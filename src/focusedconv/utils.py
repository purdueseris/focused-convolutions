import copy
import torch.nn as nn
import torch
from focusedconv import FocusedConv2d, AoIGenerator
from matplotlib import pyplot as plt
from skimage.morphology import convex_hull_image

def build_focused_model(m: nn.Module, top_layers: nn.Module, activation_brightness_threshold: float, remaining_layers: nn.Module, aoi_mask_holder: dict):
    aoi_generator = AoIGenerator(aoi_mask_holder, activation_brightness_threshold)
    focusify_all_conv2d(remaining_layers, aoi_mask_holder)
    focused_features = nn.Sequential(top_layers, aoi_generator, remaining_layers)
    m.features = focused_features
    return m

def render_aoi_mask(aoi_mask_holder: dict):
    aoi_features = aoi_mask_holder["aoi_mask"]
    plt.imshow(convex_hull_image(aoi_features))
    

def focusify_all_conv2d(m: nn.Module, aoi_mask_holder: dict):
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
            new_conv = FocusedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, aoi_mask_holder=aoi_mask_holder)
            new_conv.weight = copy.deepcopy(child_m.weight)
            new_conv.bias = copy.deepcopy(child_m.bias)

            m._modules[child_name] = new_conv
        else:
            focusify_all_conv2d(child_m, aoi_mask_holder)