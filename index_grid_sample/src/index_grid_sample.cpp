
#include <torch/extension.h>

#include "index_grid_sample.h"

namespace at::native {
Tensor index_grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid, const Tensor& index,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {grid_size[0], in_size[1]}, input.options());
  launch_index_grid_sampler_2d_forward_kernel(
      output, input, grid, index, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor index_grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid, const Tensor& index,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {grid_size[0], in_size[1]},
      input.options());
  launch_index_grid_sampler_3d_forward_kernel(
      output, input, grid, index, interpolation_mode, padding_mode, align_corners);
  return output;
}

std::tuple<Tensor, Tensor>
index_grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, const Tensor& index, int64_t interpolation_mode, int64_t padding_mode,
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
  launch_index_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, index, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<Tensor, Tensor>
index_grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, const Tensor& index, int64_t interpolation_mode, int64_t padding_mode,
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
  launch_index_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, index, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

void segment_to_indices(const Tensor& segments, const int64_t N, const int64_t M, Tensor &indices) {
  launch_segment_to_indices_kernel(segments, N, M, indices);
}

} // namespace at::native


// bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_grid_sampler_2d_cuda", &at::native::index_grid_sampler_2d_cuda);
  m.def("index_grid_sampler_3d_cuda", &at::native::index_grid_sampler_3d_cuda);
  m.def("index_grid_sampler_2d_backward_cuda", &at::native::index_grid_sampler_2d_backward_cuda);
  m.def("index_grid_sampler_3d_backward_cuda", &at::native::index_grid_sampler_3d_backward_cuda);
  m.def("segment_to_indices", &at::native::segment_to_indices);
}

