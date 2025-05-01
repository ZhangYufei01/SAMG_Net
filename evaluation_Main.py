from typing import Tuple, Union, List, Optional
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
import multiprocessing
import numpy as np
from evaluation.export_json import recursive_fix_for_json_export
default_num_processes = 8
import os
import json
from copy import deepcopy
import nibabel as nib
import math
from related_to_software.simpleitk_reader_writer import SimpleITKIO

def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]
def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn
def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: SimpleITKIO,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = 1.0
            results['metrics'][r]['IoU'] = 1.0
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)
def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: SimpleITKIO,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              file_ending: str='.nii.gz',
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    #print(file_ending)
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    #print(files_pred)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    '''if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"'''
    num_files_pred = len(files_pred)
    num_files_ref = len(files_ref)
    
    # 计算文件的交集
    files_intersection = set(files_pred).intersection(files_ref)
    num_files_intersection = len(files_intersection)
    
    print(f"文件夹 {folder_ref} 中的文件数: {num_files_ref}")
    print(f"文件夹 {folder_pred} 中的文件数: {num_files_pred}")
    print(f"文件交集中的文件数: {num_files_intersection}")
    #files_ref = [join(folder_ref, i) for i in files_pred]
    #files_pred = [join(folder_pred, i) for i in files_pred]
    files_ref = [folder_ref+'/'+i for i in files_intersection]
    files_pred = [folder_pred+'/'+i for i in files_intersection]
    #print(files_pred)
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)
    
    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    print(foreground_mean)
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
        
#if __name__ == '__main__':
#    folder_ref = '/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/pancreas/data/labelsTs'
#    folder_pred = '/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/pancreas/result/se_nnu_fold_0/predict/500'

#    output_file = '/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/pancreas/result/se_nnu_fold_0/predict/500/comparison_results.json'
#    image_reader_writer = SimpleITKIO()
#    file_ending = '.nii.gz'
#    regions = labels_to_list_of_regions([1])
#    ignore_label = None
#    num_processes = 3
#    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, regions, file_ending,  ignore_label,num_processes)
    #compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, regions,file_ending,  ignore_label,num_processes)
    #results = compute_metrics_for_folders(folder_ref, folder_pred, regions, ignore_label)
    #save_summary_json(results, output_file)

# 示例调用
#folder1 = '/dssg/home/acct-clsyzs/clsyzs-zhyf/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset104_heart/labelsTs'
#folder2 = '/dssg/home/acct-clsyzs/clsyzs-zhyf/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset104_heart/inferTs'
#labels_or_regions = [1,2,3]  # 示例标签或区域列表
#ignore_label = None  # 忽略的标签（如果有）

#results = compute_metrics_for_folders(folder1, folder2, labels_or_regions, ignore_label)
#print(results)
#save_summary_json(results, '/dssg/home/acct-clsyzs/clsyzs-zhyf/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset104_heart/inferTs/comparison_results.json')
