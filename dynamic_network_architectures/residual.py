from typing import Tuple, List, Union, Type
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.regularization import DropPath, SqueezeExcite
import numpy as np


class BasicBlockD(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # todo wideresnet?
                 ):
        """
        This implementation follows ResNet-D:

        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv

        :param conv_op:
        :param input_channels:
        :param output_channels:
        :param kernel_size: refers only to convs in feature extraction path, not to 1x1x1 conv in skip
        :param stride: only applies to first conv (and skip). Second conv always has stride 1
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op: only the first conv can have dropout. The second never has
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param stochastic_depth_p:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip


