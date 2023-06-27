import os
import sys
import torch
import torch.utils.cpp_extension


# ----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None


def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == "nt":

        def find_cl_path():
            import glob

            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64"
                        % edition
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError(
                    "Could not locate a supported Microsoft Visual C++ installation"
                )
            os.environ["PATH"] += ";" + cl_path

    # Compiler options.
    cuda_flags = [
        "-O3",
        "-std=c++14",
    ]

    # Linker options.
    if os.name == "posix":
        c_flags = ["-O3", "-std=c++14"]
        ldflags = ["-lcuda", "-lnvrtc"]
    elif os.name == "nt":
        c_flags = [
            "/O2",
            "/std=c++14",
        ]
        ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]

    # List of sources.
    source_files = [
        "src/index_grid_sample.cu",
        "src/index_grid_sample.cpp",
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(
            torch.utils.cpp_extension._get_build_directory(
                "index_grid_sample_plugin", False
            ),
            "lock",
        )
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="index_grid_sample_plugin",
        sources=source_paths,
        extra_cflags=c_flags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=True,
    )

    # Import, cache, and return the compiled module.
    import index_grid_sample_plugin

    _cached_plugin = index_grid_sample_plugin
    return _cached_plugin


class _index_grid_sample_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, indices, mode, padding_mode, align_corners):

        out = _get_plugin().index_grid_sampler_2d_cuda(
            input, grid, indices, mode, padding_mode, align_corners
        )

        ctx.save_for_backward(input, grid, indices)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return out

    @staticmethod
    def backward(ctx, grad_output):

        input, grid, indices = ctx.saved_variables
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])

        grad_input, grad_grid = _get_plugin().index_grid_sampler_2d_backward_cuda(
            grad_output,
            input,
            grid,
            indices,
            mode,
            padding_mode,
            align_corners,
            output_mask,
        )

        return grad_input, grad_grid, None, None, None, None


class _index_grid_sample_3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, indices, mode, padding_mode, align_corners):

        out = _get_plugin().index_grid_sampler_3d_cuda(
            input, grid, indices, mode, padding_mode, align_corners
        )

        ctx.save_for_backward(input, grid, indices)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return out

    @staticmethod
    def backward(ctx, grad_output):

        input, grid, indices = ctx.saved_variables
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])

        grad_input, grad_grid = _get_plugin().index_grid_sampler_3d_backward_cuda(
            grad_output,
            input,
            grid,
            indices,
            mode,
            padding_mode,
            align_corners,
            output_mask,
        )

        return grad_input, grad_grid, None, None, None, None


def index_grid_sample_2d(
    input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True
):
    # input: [B, C, H, W], float
    # grid: [..., 2], float
    # indices: [..., 1], long, the batch id for each point in grid
    # return: [..., C]

    mode = ["bilinear", "nearest", "bicubic"].index(mode)
    padding_mode = ["zeros", "border", "reflection"].index(padding_mode)

    # assert indices.max() < input.shape[0]

    prefix = grid.shape[:-1]
    grid = grid.contiguous().view(-1, 2)
    indices = indices.long().contiguous().view(-1, 1)

    output = _index_grid_sample_2d.apply(
        input, grid, indices, mode, padding_mode, align_corners
    )

    output = output.view(*prefix, input.shape[1])

    return output


def index_grid_sample_3d(
    input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True
):

    mode = ["bilinear", "nearest", "bicubic"].index(mode)
    padding_mode = ["zeros", "border", "reflection"].index(padding_mode)

    # assert indices.max() < input.shape[0]

    prefix = grid.shape[:-1]
    grid = grid.contiguous().view(-1, 3)
    indices = indices.long().contiguous().view(-1, 1)

    output = _index_grid_sample_3d.apply(
        input, grid, indices, mode, padding_mode, align_corners
    )

    output = output.view(*prefix, input.shape[1])

    return output


def index_grid_sample(
    input, grid, indices, mode="bilinear", padding_mode="zeros", align_corners=True
):
    if grid.shape[-1] == 2:
        return index_grid_sample_2d(
            input, grid, indices, mode, padding_mode, align_corners
        )
    elif grid.shape[-1] == 3:
        return index_grid_sample_3d(
            input, grid, indices, mode, padding_mode, align_corners
        )
    else:
        raise NotImplementedError(
            "Only 2D and 3D are supported for index_grid_sample()!"
        )


class _segment_to_indices(torch.autograd.Function):
    @staticmethod
    def forward(ctx, segment, max_index=None):
        # segment: [N, 2], long, each represents a segment by (offset, count)
        # max_index: max index value, if None, will be inferred from segment

        segment = segment.contiguous()
        N = segment.shape[0]

        if max_index is None:
            max_index = segment.sum(dim=1).max().item()

        indices = torch.zeros(max_index, dtype=torch.long, device=segment.device)

        _get_plugin().segment_to_indices(segment, N, max_index, indices)

        return indices


def segment_to_indices(segment, max_index=None):
    return _segment_to_indices.apply(segment, max_index)
