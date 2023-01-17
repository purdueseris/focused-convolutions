import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from skimage import morphology

class AoIGenerator(nn.Module):
    def __init__(self, aoi_mask_holder: dict, threshold: float) -> None:
        super().__init__()
        self.aoi_mask_holder = aoi_mask_holder
        self.threshold = threshold
    
    def forward(self, x:torch.Tensor):
        self.aoi_mask_holder["aoi_mask"] = torch.sum(x, dim=1).squeeze(0) >= self.threshold
        return x

def render_aoi_mask(aoi_mask_holder: dict):
    aoi_features = aoi_mask_holder["aoi_mask"]
    plt.imshow(morphology.convex_hull_image(aoi_features))

class AoISizeEstimator(nn.Module):
    def __init__(self, aoi_mask_holder, threshold: float, dataset_tracker: list, output_shape: tuple) -> None:
        super().__init__()
        
        self.aoi_mask_holder = aoi_mask_holder
        self.threshold = threshold
        self.dataset_tracker = dataset_tracker
        self.output_shape = output_shape
    
    def forward(self, x:torch.Tensor):
        self.aoi_mask_holder["aoi_mask"] = torch.sum(x, dim=1).squeeze(0) >= self.threshold
        aoi_nz_coords = torch.nonzero(self.aoi_mask_holder["aoi_mask"])
        if aoi_nz_coords.numel() == 0:
           self.dataset_tracker.append(1.0)
           return torch.zeros(self.output_shape).to(x.device)

        h_in = x.shape[2]
        w_in = x.shape[3]
        aoi_y1 = torch.min(aoi_nz_coords[:, 0])
        aoi_y2 = torch.max(aoi_nz_coords[:, 0])+1
        aoi_x1 = torch.min(aoi_nz_coords[:, 1])
        aoi_x2 = torch.max(aoi_nz_coords[:, 1])+1
        self.dataset_tracker.append((aoi_y2-aoi_y1)*(aoi_x2-aoi_x1)/(h_in*w_in))
        return torch.zeros(self.output_shape, device=x.device)