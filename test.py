import torch
import torch.nn.functional as F

from segment_grid_sample import segment_grid_sample_2d, segment_grid_sample_3d

# 2d 
input = torch.rand(2, 3, 128, 128, dtype=torch.float32, device='cuda')
grid = torch.rand(2, 1, 8, 2, dtype=torch.float32, device='cuda') * 2 - 1 # in [-1, 1]

gt = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
pred = segment_grid_sample_2d(input, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

print(torch.allclose(gt, pred))