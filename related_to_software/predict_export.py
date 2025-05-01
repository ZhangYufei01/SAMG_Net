import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
##from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union
default_num_processes = 8
from related_to_software.simpleitk_reader_writer import SimpleITKIO

def pad_bbox(bounding_box: Union[List[List[int]], Tuple[Tuple[int, int]]], pad_amount: Union[int, List[int]],
             array_shape: Tuple[int, ...] = None) -> List[List[int]]:
    """

    """
    if isinstance(bounding_box, tuple):
        # convert to list
        bounding_box = [list(i) for i in bounding_box]
    else:
        # needed because otherwise we overwrite input which could have unforseen consequences
        bounding_box = deepcopy(bounding_box)

    if isinstance(pad_amount, int):
        pad_amount = [pad_amount] * len(bounding_box)

    for i in range(len(bounding_box)):
        new_values = [max(0, bounding_box[i][0] - pad_amount[i]), bounding_box[i][1] + pad_amount[i]]
        if array_shape is not None:
            new_values[1] = min(array_shape[i], new_values[1])
        bounding_box[i] = new_values

    return bounding_box


def regionprops_bbox_to_proper_bbox(regionprops_bbox: Tuple[int, ...]) -> List[List[int]]:
    """
    regionprops_bbox is what you get from `from skimage.measure import regionprops`
    """
    dim = len(regionprops_bbox) // 2
    return [[regionprops_bbox[i], regionprops_bbox[i + dim]] for i in range(dim)]


def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])


def crop_to_bbox(array: np.ndarray, bounding_box: List[List[int]]):
    assert len(bounding_box) == len(array.shape), f"Dimensionality of bbox and array do not match. bbox has length " \
                                          f"{len(bounding_box)} while array has dimension {len(array.shape)}"
    slicer = bounding_box_to_slice(bounding_box)
    return array[slicer]


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!

    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]

def convert_probabilities_to_segmentation(predicted_probabilities: Union[np.ndarray, torch.Tensor],has_regions = False,regions_class_order:int=2,num_segmentation_heads:int=2) -> \
        Union[np.ndarray, torch.Tensor]:
    """
    assumes that inference_nonlinearity was already applied!

    predicted_probabilities has to have shape (c, x, y(, z)) where c is the number of classes/regions
    """
    if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
        raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
                           f" got {type(predicted_probabilities)}")

    if has_regions:
        assert regions_class_order is not None, 'if region-based training is requested then you need to ' \
                                                     'define regions_class_order!'
        # check correct number of outputs
    assert predicted_probabilities.shape[0] == num_segmentation_heads, \
        f'unexpected number of channels in predicted_probabilities. Expected {num_segmentation_heads}, ' \
        f'got {predicted_probabilities.shape[0]}. Remember that predicted_probabilities should have shape ' \
        f'(c, x, y(, z)).'

    if has_regions:
        if isinstance(predicted_probabilities, np.ndarray):
            segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.uint16)
        else:
            # no uint16 in torch
            segmentation = torch.zeros(predicted_probabilities.shape[1:], dtype=torch.int16,
                                       device=predicted_probabilities.device)
        for i, c in enumerate(regions_class_order):
            segmentation[predicted_probabilities[i] > 0.5] = c
    else:
        segmentation = predicted_probabilities.argmax(0)

    return segmentation

def custom_sigmoid(x):
    return 1 / (1 + np.exp(-x))

#核心计算在convert_probabilities_to_segmentation
def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if isinstance(predicted_logits, np.ndarray):
        predicted_logits = torch.from_numpy(predicted_logits)
    #predicted_logits = predicted_logits.to(torch.float16)
    predicted_probabilities = custom_sigmoid(predicted_logits)
    del predicted_logits
    segmentation = convert_probabilities_to_segmentation(predicted_probabilities,has_regions = False,regions_class_order=2,num_segmentation_heads=2)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    #segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],dtype=np.uint8)
    #slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    #segmentation_reverted_cropping[slicer] = segmentation
    segmentation_reverted_cropping = segmentation
    del segmentation

    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping

#主要作用是调用convert_predicted_logits_to_segmentation_with_correct_shape
def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  ofile:str,save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)


    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(ofile[:-7] + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, ofile[:-7] + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    SimpleITKIO().write_seg(segmentation_final, ofile, properties_dict)



