import os
import sys
import torch
import torch.utils.cpp_extension


#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = []

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'src/segment_grid_sample.cu',
        'src/segment_grid_sample.cpp',
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('segment_grid_sample_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='segment_grid_sample_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import segment_grid_sample_plugin
    _cached_plugin = segment_grid_sample_plugin
    return _cached_plugin

class _segment_grid_sample_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, interpolation_mode, padding_mode, align_corners):

        out = _get_plugin().segment_grid_sampler_2d_cuda(input, grid, interpolation_mode, padding_mode, align_corners)

        ctx.save_for_backward(input, grid)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return out

    @staticmethod
    def backward(ctx, grad_output):
         
        input, grid = ctx.saved_variables
        interpolation_mode = ctx.interpolation_mode 
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])

        grad_input, grad_grid = _get_plugin().segment_grid_sampler_2d_backward_cuda(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask)
    
        return grad_input, grad_grid, None, None, None


def segment_grid_sample_2d(input, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True):
    # input: [N, 3]

    interpolation_mode = ['bilinear', 'nearest', 'bicubic'].index(interpolation_mode)
    padding_mode = ['zeros', 'border', 'reflection'].index(padding_mode)    

    return _segment_grid_sample_2d.apply(input, grid, interpolation_mode, padding_mode, align_corners)


class _segment_grid_sample_3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, interpolation_mode, padding_mode, align_corners):

        out = _get_plugin().segment_grid_sampler_3d_cuda(input, grid, interpolation_mode, padding_mode, align_corners)

        ctx.save_for_backward(input, grid)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return out

    @staticmethod
    def backward(ctx, grad_output):
         
        input, grid = ctx.saved_variables
        interpolation_mode = ctx.interpolation_mode 
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])

        grad_input, grad_grid = _get_plugin().segment_grid_sampler_3d_backward_cuda(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask)
    
        return grad_input, grad_grid, None, None, None


def segment_grid_sample_3d(input, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True):
    # input: [N, 3]

    interpolation_mode = ['bilinear', 'nearest', 'bicubic'].index(interpolation_mode)
    padding_mode = ['zeros', 'border', 'reflection'].index(padding_mode)    

    return _segment_grid_sample_3d.apply(input, grid, interpolation_mode, padding_mode, align_corners)