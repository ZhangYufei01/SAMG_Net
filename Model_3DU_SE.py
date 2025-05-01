import torch
import torch.nn as nn
from typing import Union, Type, List, Tuple
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = [(i - 1) // 2 for i in kernel_size]
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=1,bias=True)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.all_modules = nn.Sequential(self.conv, self.norm, self.nonlin)

    def forward(self, x):
        return self.all_modules(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DoubleConv, self).__init__()
        self.convs = nn.Sequential(
            ConvBlock(
                in_channels, out_channels, kernel_size, stride
            ),
            ConvBlock(
                    out_channels, out_channels, kernel_size, 1
            )
        )
    def forward(self, x):
        return self.convs(x)

class SE_Block3D(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SE_Block3D, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Encoder(nn.Module):
    def __init__(self,input_channels, n_stages, features_per_stage, kernel_sizes, strides, return_skips: bool = True):
        super(Encoder, self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        stages = []
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            stage_modules.append(DoubleConv(
                input_channels, features_per_stage[s], kernel_sizes[s], conv_stride
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.return_skips = return_skips
    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class Decoder(nn.Module):
    def __init__(self,features_per_stage, kernel_sizes, strides, output_clannels):
        super(Decoder, self).__init__()

        n_stages_encoder = len(features_per_stage)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * (n_stages_encoder - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * (n_stages_encoder - 1)
        if isinstance(strides, int):
            strides = [strides] * (n_stages_encoder - 1)

        stages = []
        transpconvs = []
        seg_layers = []
        se_blocks = []
        for s in range(1, n_stages_encoder):
            input_features_below = features_per_stage[-s]
            input_features_skip = features_per_stage[-(s + 1)]
            stride_for_transpconv = strides[-s]

            transpconvs.append(nn.ConvTranspose3d(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,bias=True
            ))

            stages.append(DoubleConv(
                2 * input_features_skip, input_features_skip,kernel_sizes[-(s + 1)], 1
            ))

            se_blocks.append(SE_Block3D(input_features_skip))
            seg_layers.append(nn.Conv3d(input_features_skip, output_clannels, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.se_blocks = nn.ModuleList(se_blocks)

    def forward(self, skips):

        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if s == (len(self.stages) - 1):
                #print('lin127',x.shape)
                x = self.se_blocks[s](x)
                #print('lin129', x.shape)

                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        r = seg_outputs[0]
        return r

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_stages=6,
                 features_per_stage= [32, 64, 128, 256, 320, 320],
                 kernel_sizes= [[3, 3, 3]] * 6,
                 strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        self.encoder = Encoder(in_channels, n_stages, features_per_stage, kernel_sizes, strides)
        self.decoder = Decoder(features_per_stage, kernel_sizes,strides,out_channels)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

#network = UNet(in_channels=2, out_channels=2)
#x=torch.randn((1, 2, 96, 128, 128))
#y = network(x)
#print(y.shape)

