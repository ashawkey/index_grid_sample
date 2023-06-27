
#include <torch/extension.h>

#include "segment_grid_sample.h"

namespace at::native {
Tensor segment_grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_segment_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor segment_grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_segment_grid_sampler_3d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

std::tuple<Tensor, Tensor>
segment_grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_segment_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<Tensor, Tensor>
segment_grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool,2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_segment_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}
}

// bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_grid_sampler_2d_cuda", &at::native::segment_grid_sampler_2d_cuda);
  m.def("segment_grid_sampler_3d_cuda", &at::native::segment_grid_sampler_3d_cuda);
  m.def("segment_grid_sampler_2d_backward_cuda", &at::native::segment_grid_sampler_2d_backward_cuda);
  m.def("segment_grid_sampler_3d_backward_cuda", &at::native::segment_grid_sampler_3d_backward_cuda);
}

