from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass

class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def normalize_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)

        # 线性缩放将像素值映射到 [0, 1]
        normalized_image = (image - min_val) / (max_val - min_val)

        return normalized_image

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)

        #这一步是我自己加来限制输出范围到0-1的
        normalized_image = self.normalize_image(image)
        #normalized_image = image
        return normalized_image