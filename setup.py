#!/usr/bin/env python

import glob
import os
from os import path
from setuptools import find_packages, setup, Extension

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "lib", "ops", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "lib.ops._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="Pet",
    version='0.4c',
    license='MIT',
    author="BUPT-PRIV",
    url="https://github.com/BUPT-PRIV/Pet",
    description="Pytorch Efficient Toolbox (Pet) for Computer Vision.",
    packages=find_packages(exclude=("cfgs", "ckpts", "data", "weights")),
    python_requires=">=3.5",
    install_requires=[
        "termcolor>=1.1",
        "Pillow",  # you can also use pillow-simd for better performance
        "yacs>=0.1.6",
        "pyyaml",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "numpy",
        "opencv-python>=3.4.0",
        "scipy",
        "six",
        "h5py",
        "scikit-image",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
