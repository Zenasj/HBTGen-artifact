import torch
device = torch.device('cuda')
# works very well, no error occured .

import os
import re
import sys
import platform
import subprocess
import torch

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable
                      # '-DProtobuf_DIR=.../tmp_install/cmake'
        ]

        #if hasattr(torch.utils, "cmake_prefix_path"):
        #    cmake_args.append('-DCMAKE_PREFIX_PATH=' + torch.utils.cmake_prefix_path)
        #    print(f'torch cmake:: {torch.utils.cmake_prefix_path}')
        cmake_args.append('-DCMAKE_PREFIX_PATH=' + '/workspace/work_dir/libtorch_gpu/libtorch')
        # cmake_args.append('-DProtobuf_DIR=' + '/opt/conda/envs/final_01/lib/python3.8/site-packages/torch/lib/tmp_install/cmake')
        # /opt/conda/envs/final_01/lib/python3.8/site-packages/torch/lib/tmp_install
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', '--std=c++17'),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='some_name',
    version='0.0.4',
    author='some_named',
    description='some library',
    keywords='some keywords',
    long_description='some library',
    ext_modules=[CMakeExtension('target_name')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)