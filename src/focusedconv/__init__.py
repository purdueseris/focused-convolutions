"""A package for focused convolutions"""

import torch.nn as nn
import copy
from .areaofinterest import AoIThresholder, AoI
from .convolution import FocusedConv2d


def build_focused_model(m: nn.Module, top_layers: nn.Module, activation_brightness_threshold: float, remaining_layers: nn.Module):
    aoi = AoI()
    aoi_generator = AoIThresholder(activation_brightness_threshold, aoi)
    focusify_all_conv2d(remaining_layers, aoi)
    focused_features = nn.Sequential(top_layers, aoi_generator, remaining_layers)
    m.features = focused_features
    return m, aoi

def focusify_all_conv2d(m: nn.Module, aoi: AoI):
    for child_name in m._modules:
        child_m = m._modules[child_name]
        if type(child_m) == nn.Conv2d:
            in_channels = child_m.in_channels
            out_channels = child_m.out_channels
            kernel_size = child_m.kernel_size
            stride = child_m.stride
            padding = child_m.padding
            dilation = child_m.dilation
            groups = child_m.groups
            new_conv = FocusedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, aoi=aoi)
            new_conv.weight = copy.deepcopy(child_m.weight)
            new_conv.bias = copy.deepcopy(child_m.bias)

            m._modules[child_name] = new_conv
        else:
            focusify_all_conv2d(child_m, aoi)
    