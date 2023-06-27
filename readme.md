# Index-grid-sample

Extension to `F.grid_sample` that allows using batch index for each input grid point.

The APIs are the same as `F.grid_sample` other than:
* `grid` can be in arbitrary shape `[..., 2/3]` except for the last dimension (must be in 2D or 3D).
* An extra `indices` input of `[..., 1] (torch.long)` is required after `grid` to specify the batch indices.

### Install

Assume `torch` already installed.

```bash
pip install git+https://github.com/ashawkey/index_grid_sample

# or locally
git clone https://github.com/ashawkey/index_grid_sample
cd index_grid_sample
pip install .
```

### Usage

```python

import torch

from index_grid_sample import index_grid_sample, segments_to_indices

B = 2 # batch size of input
C = 3 # feature channels
N = 512 # number of sample points

### 2D exmaple
input = torch.rand(B, C, 1024, 1024, dtype=torch.float32, device="cuda")
grid = torch.rand(N, 2, dtype=torch.float32, device="cuda") * 2 - 1
indices = torch.randint(0, B, (N, 1), dtype=torch.long, device="cuda")

# [N, C]
results = index_grid_sample(input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True)


### we also provide a helper function to convert segments to indices
# segments are defined as [B, 2], each entry is (offset, length) per batch
segments = torch.tensor([[0, 3], [5, 1], [3, 2]], dtype=torch.long, device='cuda')
indices = segments_to_indices(segments)
# indices == [0, 0, 0, 2, 2, 1]
```