import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from skimage import morphology

class AoI():
    def __init__(self, mask_tensor=torch.Tensor()) -> None:
        self.mask_tensor = mask_tensor
    
    def imshow(self):
        plt.imshow(morphology.convex_hull_image(self.mask_tensor))

    def calculate_aoi_size(self):
        aoi_nz_coords = torch.nonzero(self.mask_tensor)
        if aoi_nz_coords.numel() == 0:
           return 1.0
        h_in = self.mask_tensor.shape[0]
        w_in = self.mask_tensor.shape[1]
        aoi_y1 = torch.min(aoi_nz_coords[:, 0])
        aoi_y2 = torch.max(aoi_nz_coords[:, 0])+1
        aoi_x1 = torch.min(aoi_nz_coords[:, 1])
        aoi_x2 = torch.max(aoi_nz_coords[:, 1])+1
        return float((aoi_y2-aoi_y1)*(aoi_x2-aoi_x1)/(h_in*w_in))


class AoIThresholder(nn.Module):
    def __init__(self, threshold: float, aoi: AoI=AoI()) -> None:
        super().__init__()
        self.aoi = aoi
        self.threshold = threshold
    
    def forward(self, x:torch.Tensor):
        self.aoi.mask_tensor = (torch.sum(x, dim=1).squeeze(0) >= self.threshold)
        return x