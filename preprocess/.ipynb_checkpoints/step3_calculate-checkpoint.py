import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

from related_to_software.preprocess_cropping import crop_to_nonzero
from related_to_software.simpleitk_reader_writer import SimpleITKIO

class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], semi: Union[np.ndarray, None], properties: dict,intensityproperties:dict):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)
            
        if semi is not None:
            assert data.shape[1:] == semi.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            semi = np.copy(semi)

        has_seg = seg is not None

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, semi, bbox = crop_to_nonzero(data, seg, semi)
        #print(bbox)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, intensityproperties)
        #data = self._normalize(data, seg)
        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            unique_chars = np.unique(seg).tolist()
            collect_for_this = [x for x in unique_chars if x != 0.0]
            #print(collect_for_this)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, semi

    def run_case(self, image_files: List[str], seg_file: Union[str, None], semi_file: Union[str, None],intensityproperties):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """

        rw = SimpleITKIO()

        # load image(s)
        #print('before itk',image_files)
        data, data_properties = rw.read_images([image_files])
        #print(data.shape)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg([seg_file])
        else:
            seg = None
            
        if semi_file is not None:
            semi, _ = rw.read_seg([semi_file])
        else:
            semi = None

        data, seg, semi = self.run_case_npy(data, seg, semi, data_properties, intensityproperties)
        return data, seg, semi, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,semi_files:str,intensityproperties:dict):

        key = image_files.split("/")[-1][:-7]
        #print(output_filename_truncated+'/'+key)
        data, seg, semi, properties = self.run_case(image_files, seg_file, semi_files, intensityproperties)
        #print('in line 99:',data.shape)
        print(output_filename_truncated,key)
        np.savez_compressed(output_filename_truncated+'/'+key + '.npz', image=data, semi_auto=semi, label=seg)
        write_pickle(properties, output_filename_truncated+'/'+key + '.pkl')
        npz_path = output_filename_truncated+'/'+key + '.npz'
        pkl_path = output_filename_truncated+'/'+key + '.pkl'
        return key,npz_path,pkl_path

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs



    def _normalize(self, data: np.ndarray, seg: np.ndarray,intensityproperties) -> np.ndarray:

        mean_intensity = intensityproperties['mean']
        std_intensity = intensityproperties['std']
        lower_bound = intensityproperties['percentile_00_5']
        upper_bound = intensityproperties['percentile_99_5']

        np.clip(data, lower_bound, upper_bound, out=data)
        data -= mean_intensity
        data /= max(std_intensity, 1e-8)
        return data




    def run(self, dataset_json,output_folder,preprocess_json,
            num_processes: int):

        if not isdir(output_folder):
            maybe_mkdir_p(output_folder)

        with open(dataset_json, 'r') as f:
            config = json.load(f)
        dataset = config['data']
        intensityproperties = config['intensity_statistics_per_channel']
        #print(dataset)
        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        json_data = {
            'keys': [],
            'data_paths': [],
            'pkl_paths': []
            }
        for k in dataset:
            #print(output_folder, k['image'], k['label'], k['semi_auto_infor'])
            key,npz_path,pkl_path =self.run_case_save(output_folder, k['image'], k['label'], k['semi_auto_infor'],intensityproperties)
            print(key)
            json_data['keys'].append(key)
            json_data['data_paths'].append(npz_path)
            json_data['pkl_paths'].append(pkl_path)
        
        json_file_path = preprocess_json  # 替换为你要保存的 JSON 文件路径
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)

    def modify_seg_fn(self, seg: np.ndarray) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg




#pp = DefaultPreprocessor()
#pp.run('/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/data/data_summary.json', '/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/data/preprocess_data',"/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/data/data_preprocess.json",3)
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
