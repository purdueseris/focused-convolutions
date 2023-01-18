import math
import torch.nn as nn
import torch.nn.functional as F
import torch

from enum import Enum
from torch.nn.common_types import _size_2_t
from typing import Union
from .areaofinterest import AoI

class FOCUSED_CONV_OP_MODES(Enum):
    AOI_GENERAL_MODE = 0
    AOI_RECT_MODE = 1


    
    
        
class FocusedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        aoi: AoI=None,
        focused_conv_op_mode=FOCUSED_CONV_OP_MODES.AOI_RECT_MODE
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        self.aoi = aoi
        self.focused_conv_op_mode = focused_conv_op_mode
        self.weight_mat = self.weight.view(self.out_channels, -1) # size outchannels X patchlen

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        h_in = x.shape[2]
        w_in = x.shape[3]

        # H, W out for Conv2d
        h_out = math.floor(
            (h_in + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0]+1)
        w_out = math.floor(
            (w_in + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1]+1)

        if self.focused_conv_op_mode == FOCUSED_CONV_OP_MODES.AOI_GENERAL_MODE:
            aoi_patches = F.unfold(self.aoi.mask_tensor, kernel_size=self.kernel_size, padding=self.padding) # size 1 x patchLen x numpatches
            patch_idxes_to_keep = aoi_patches.sum(dim=1).squeeze(0) > 0 # size numpatches
            self.out_mat = torch.zeros([b, self.out_channels, len(patch_idxes_to_keep)], device=x.device) # size batchsize X outchannels X numpatches
            in_patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding) # size batchsize X patchlen X numpatches
            selected_in_patches = in_patches[:, :, patch_idxes_to_keep] # size batchsize X patchlen X num patches selected
            selected_out_mat = torch.matmul(self.weight_mat, selected_in_patches)
            self.out_mat[:, :, patch_idxes_to_keep] = selected_out_mat
            out = self.out_mat.view(b, self.out_channels, h_out, -1) # size batchsize X outchannels X heightOut X widthOut
        
        elif self.focused_conv_op_mode == FOCUSED_CONV_OP_MODES.AOI_RECT_MODE:
            h_mask = self.aoi.mask_tensor.shape[0]
            w_mask = self.aoi.mask_tensor.shape[1]

            aoi_nz_coords = torch.nonzero(self.aoi.mask_tensor)
            if aoi_nz_coords.numel() == 0:
                out = F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups
                )
                return out

            # out Tensors
            out = torch.zeros([b, self.out_channels, h_out, w_out], device=x.device)

            aoi_y1 = torch.min(aoi_nz_coords[:, 0])
            aoi_y2 = torch.max(aoi_nz_coords[:, 0])+1
            aoi_x1 = torch.min(aoi_nz_coords[:, 1])
            aoi_x2 = torch.max(aoi_nz_coords[:, 1])+1

            in_x1 = math.floor(aoi_x1*w_in/w_mask)
            in_x2 = math.floor(aoi_x2*w_in/w_mask)
            in_y1 = math.floor(aoi_y1*h_in/h_mask)
            in_y2 = math.floor(aoi_y2*h_in/h_mask)

            out_x1 = math.floor(aoi_x1*w_out/w_mask)
            out_x2 = math.floor(aoi_x2*w_out/w_mask)
            out_y1 = math.floor(aoi_y1*h_out/h_mask)
            out_y2 = math.floor(aoi_y2*h_out/h_mask)

            try:
                out[:, :, out_y1:out_y2, out_x1:out_x2] = F.conv2d(
                    x[:, :, in_y1:in_y2, in_x1:in_x2],
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups
                )
            except:
                out = F.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups
                )
        else:
            out = F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

        return out
