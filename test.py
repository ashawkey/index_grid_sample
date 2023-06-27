import torch
import torch.nn.functional as F
import time

from segment_grid_sample import segment_grid_sample_2d, segment_grid_sample_3d

# 2d
input = torch.rand(2, 3, 1024, 1024, dtype=torch.float32, device="cuda")
grid = (
    torch.rand(2, 1, 1024, 2, dtype=torch.float32, device="cuda") * 2 - 1
)  # in [-1, 1]

gt = F.grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
)
pred = segment_grid_sample_2d(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
)

print(torch.allclose(gt, pred))

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
    pred = segment_grid_sample_2d(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
torch.cuda.synchronize()
t1 = time.time()
print("plugin", t1 - t0)
