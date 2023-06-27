#pragma once
#include <array>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

void launch_index_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid, const TensorBase &indices,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void launch_index_grid_sampler_3d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid, const TensorBase &indices,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void launch_index_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, const TensorBase &indices, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

void launch_index_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, const TensorBase &indices, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

void launch_segment_to_indices_kernel(
    const TensorBase &segments, const int64_t N, const int64_t M, TensorBase &indices);

}  // namespace at::native
