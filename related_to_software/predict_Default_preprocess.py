import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice
from  related_to_software.predict_normalization import CTNormalization
class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def create_nonzero_mask(self, data):
        """

        :param data:
        :return: the mask is True where the data is nonzero
        """
        from scipy.ndimage import binary_fill_holes
        assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
        nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
        for c in range(data.shape[0]):
            this_mask = data[c] != 0
            nonzero_mask = nonzero_mask | this_mask
        nonzero_mask = binary_fill_holes(nonzero_mask)
        return nonzero_mask

    def crop_to_nonzero(self, data, seg=None, nonzero_label=-1):
        """

        :param data:
        :param seg:
        :param nonzero_label: this will be written into the segmentation map
        :return:
        """
        data=np.round(data, 3)

        nonzero_mask = self.create_nonzero_mask(data)

        bbox = get_bbox_from_mask(nonzero_mask)

        slicer = bounding_box_to_slice(bbox)
        data = data[tuple([slice(None), *slicer])]

        if seg is not None:
            seg = seg[tuple([slice(None), *slicer])]

        nonzero_mask = nonzero_mask[slicer][None]
        if seg is not None:
            seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
        else:
            nonzero_mask = nonzero_mask.astype(np.int8)
            nonzero_mask[nonzero_mask == 0] = nonzero_label
            nonzero_mask[nonzero_mask > 0] = 0
            seg = nonzero_mask
        return data, seg, bbox

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel) -> np.ndarray:
        for c in range(data.shape[0]):
            #print(data.shape[0])
            #scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = CTNormalization
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{normalizer_class}\' for normalization')

            if foreground_intensity_properties_per_channel is not None:
                intensity_properties = foreground_intensity_properties_per_channel[str(c)]
            else:
                intensity_properties = {
                    'mean': np.mean(data[c]),
                    'std': np.std(data[c]),
                    'percentile_00_5': np.percentile(data[c], 0.5),
                    'percentile_99_5': np.percentile(data[c], 99.5)
                }
                #print(intensity_properties)
            normalizer = normalizer_class(use_mask_for_norm=None,
                                          intensityproperties=intensity_properties)
            if seg is not None:
                data[c] = normalizer.run(data[c], seg[0])
            else:
                data[c] = normalizer.run(data[c])
            #print(f'here',data.shape)
        return data

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     foreground_intensity_properties_per_channel:Union[dict,None]):
        #properties=properties[0]
        #print(properties)
        #print(f'in Default_pre,data.shape:{data.shape}')
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None
        #print(data.min())

        threshold_low = data.min()
        data = self._normalize(data, seg, foreground_intensity_properties_per_channel)
        #print(data)
        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        #print(shape_before_cropping)
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = self.crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]


        #print(properties)
        return data, seg, properties