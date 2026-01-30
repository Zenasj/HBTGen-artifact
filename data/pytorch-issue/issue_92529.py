import glob
import os
import shutil
import subprocess
import sys
from os import path
from setuptools import find_packages, setup
import numpy as np

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python
from setuptools import Extension
from Cython.Build import cythonize


def get_version():
    version_file = 'det/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


torch_ver = [int(x) for x in torch.__version__.split('.')[:2]]
assert torch_ver >= [1, 4], 'Requires PyTorch >= 1.4'
_NP_INCLUDE_DIRS = np.get_include()


def install_package(package):
    output = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'install', package])
    print(output.decode())


def load_package(requirements_path='requirements.txt'):
    requirements = []
    with open(requirements_path, 'r') as f:
        for each in f.readlines():
            requirements.append(each.strip())
    return requirements


def load_scripts(scripts_path: list):
    scripts = []
    for sub_path in scripts_path:
        for each_scripts in os.listdir(sub_path):
            scripts.append('{}/{}'.format(sub_path, each_scripts))

    return scripts


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, 'det', 'core', 'layer', 'csrc')

    main_source = path.join(extensions_dir, 'vision.cpp')
    sources = glob.glob(path.join(extensions_dir, '**', '*.cpp'))

    is_rocm_pytorch = False
    if torch_ver >= [1, 5]:
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (True if ((torch.version.hip is not None) and
                                    (ROCM_HOME is not None)) else False)

    if is_rocm_pytorch:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes='/det/core/layer/csrc/*',
            show_detailed=True,
            is_pytorch_extension=True,
        )

        # Current version of hipify function in pytorch creates an intermediate directory
        # named "hip" at the same level of the path hierarchy if a "cuda" directory exists,
        # or modifying the hierarchy, if it doesn't. Once pytorch supports
        # "same directory" hipification (https://github.com/pytorch/pytorch/pull/40523),
        # the source_cuda will be set similarly in both cuda and hip paths, and the explicit
        # header file copy (below) will not be needed.
        source_cuda = glob.glob(
            path.join(extensions_dir, '**', 'hip', '*.hip')) + glob.glob(
                path.join(extensions_dir, 'hip', '*.hip'))

        shutil.copy(
            'det/core/layer/csrc/box_iou_rotated/box_iou_rotated_utils.h',
            'det/core/layer/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h',
        )
        shutil.copy(
            'det/core/layer/csrc/deformable/deform_conv.h',
            'det/core/layer/csrc/deformable/hip/deform_conv.h',
        )

    else:
        source_cuda = glob.glob(path.join(
            extensions_dir, '**', '*.cu')) + glob.glob(
                path.join(extensions_dir, '*.cu'))

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {'cxx': []}
    define_macros = []

    if (torch.cuda.is_available() and
        ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
            'FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [('WITH_CUDA', None)]
            extra_compile_args['nvcc'] = [
                '-O3',
                '-DCUDA_HAS_FP16=1',
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        else:
            define_macros += [('WITH_HIP', None)]
            extra_compile_args['nvcc'] = []

        # It's better if pytorch can do this by default ..
        CC = os.environ.get('CC', None)
        if CC is not None:
            extra_compile_args['nvcc'].append('-ccbin={}'.format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'det._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ]
    ext_modules_cython = cythonize([
        Extension(
            name='det.model.post_processor.cython_nms',
            sources=['det/model/post_processor/cython_nms.pyx'],
            extra_compile_args=['-Wno-cpp'],
            include_dirs=[_NP_INCLUDE_DIRS])
    ])

    return ext_modules + ext_modules_cython


setup(
    name='det',
    version=get_version(),
    description='SMore-Det: Detection codebase.',
    packages=find_packages(exclude=('configs', 'tests*')),
    install_requires=load_package('./requirements.txt'),
    include_package_data=False,
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)