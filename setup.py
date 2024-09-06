# --------------------------------------------------------------------------------------
# Modified from :
#   https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# Inspired by :
#   https://github.com/pytorch/extension-cpp/tree/master
# --------------------------------------------------------------------------------------
# type: ignore


import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

package_name = "ms_deform_attn"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"

    if debug_mode:
        print("Compiling in debug mode")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    define_macros = []

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    curdir = os.path.curdir
    extensions_dir = os.path.join(curdir, "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    if torch.cuda.is_available() and CUDA_HOME is not None:
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise RuntimeError("CUDA is not available")

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            name=f"{package_name}._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    # disable ninja because BuildExtension will try to make sources path absolute,
    # which is incompatible with build and will raise an exception
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
