import torch
import torch.nn.functional as F
import time

from index_grid_sample import index_grid_sample, segment_to_indices

# 2d
input = torch.rand(2, 3, 1024, 1024, dtype=torch.float32, device="cuda")
grid = torch.rand(2, 1, 1024, 2, dtype=torch.float32, device="cuda") * 2 - 1

gt_indices = torch.cat(
    [
        torch.zeros(1, 1, 1024, 1, dtype=torch.long, device="cuda"),
        torch.ones(1, 1, 1024, 1, dtype=torch.long, device="cuda"),
    ],
    dim=0,
)

segments = torch.tensor([[0, 1024], [1024, 1024]], dtype=torch.long, device="cuda")
indices = segment_to_indices(segments).view(2, 1, 1024, 1)

print(torch.allclose(gt_indices, indices))

gt = F.grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
)
pred = (
    index_grid_sample(
        input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    .permute(0, 3, 1, 2)
    .contiguous()
)

print(torch.allclose(gt, pred))

# 3d
input = torch.rand(2, 4, 256, 256, 256, dtype=torch.float32, device="cuda")
grid = torch.rand(2, 1, 1, 1024, 3, dtype=torch.float32, device="cuda") * 2 - 1

indices = torch.cat(
    [
        torch.zeros(1, 1, 1, 1024, 1, dtype=torch.long, device="cuda"),
        torch.ones(1, 1, 1, 1024, 1, dtype=torch.long, device="cuda"),
    ],
    dim=0,
)

gt = F.grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
)
pred = (
    index_grid_sample(
        input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    .permute(0, 4, 1, 2, 3)
    .contiguous()
)

print(torch.allclose(gt, pred))

# simple time measurement
t0 = time.time()
for i in range(10):
    gt = F.grid_sample(
        input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
torch.cuda.synchronize()
t1 = time.time()
print("torch", t1 - t0)

t0 = time.time()
for i in range(10):
    pred = index_grid_sample(
        input,
        grid,
        indices,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
torch.cuda.synchronize()
t1 = time.time()
print("plugin", t1 - t0)
