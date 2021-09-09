# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import setuptools

path_helper_path = 'galileo/platform/path_helper.py'
if os.path.exists(path_helper_path):
    exec(open(path_helper_path).read(), globals())
else:
    from .path_helper import *

from subprocess import check_call
from distutils.command.build import build as _build
from setuptools.command.develop import develop as _develop
from setuptools.command.build_ext import build_ext as _build_ext
from distutils import log


class build(_build):
    _build.user_options.append(('skip-engine', 's', 'skip build engine'))
    _build.boolean_options.append('skip-engine')

    def initialize_options(self):
        _build.initialize_options(self)
        self.skip_engine = False

    def run(self):
        if not self.skip_engine and os.path.isdir(engine_src_dir):
            self.build_engine()
        _build.run(self)

    def _get_python_version(self):
        #for cpu image
        cpu_path = '/usr/local/bin/python3'
        # for gpu image
        gpu_path = '/usr/local/anaconda3/bin/python3'
        if os.path.exists(cpu_path):
            return f'-DPYTHON_EXECUTABLE={cpu_path}'
        elif os.path.exists(gpu_path):
            return f'-DPYTHON_EXECUTABLE={gpu_path}'
        else:
            return ''

    def build_engine(self):
        self.mkpath(engine_build_dir)
        # run cmake
        if self.force or not os.path.isfile(
                os.path.join(engine_build_dir, 'Makefile')):
            if self.debug:
                log.info('build engine debug')
                build_type = 'Debug'
            else:
                log.info('build engine release')
                build_type = 'Release'
            cmake = [
                'cmake', '-DCMAKE_BUILD_TYPE=' + build_type,
                self._get_python_version(), engine_src_dir
            ]
            check_call(cmake, cwd=engine_build_dir)
        # build
        if self.force or not is_targets_exists():
            check_call([
                'cmake', '--build', '.', '-v', '-j',
                get_max_jobs(self.parallel)
            ],
                       cwd=engine_build_dir)
        # copy targets files to pywrap
        self.mkpath(pywrap_dir)
        for t in get_cpp_targets():
            self.copy_file(t, libs_dir)
        for t in get_py_targets():
            self.copy_file(t, pywrap_dir)


class develop(_develop):
    _develop.user_options.append(('strip-so', 'r', 'strip so files'))
    _develop.boolean_options.append('strip-so')

    def initialize_options(self):
        _develop.initialize_options(self)
        self.strip_so = False

    def run(self):
        _develop.run(self)
        if self.strip_so:
            check_call([
                '/usr/bin/find', 'galileo', '-type', 'f', '-name', '*.so',
                '-exec', '/usr/bin/strip', '{}', ';'
            ],
                       cwd=project_root_dir)
            log.info('strip all so files')


class build_ext(_build_ext):
    def build_extensions(self):
        # Avoid a gcc warning
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()


def CppExtension(name, sources, *args, **kwargs):
    '''
    Creates a :class:`setuptools.Extension` for C++.

    engine includes and librarys
    '''
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.append(project_root_dir)
    include_dirs.append(engine_src_dir)
    include_dirs.append(engine_build_dir)
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.append(pywrap_dir)
    library_dirs.append(libs_dir)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('client')
    kwargs['libraries'] = libraries

    runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
    runtime_library_dirs.append(pywrap_dir)
    runtime_library_dirs.append(libs_dir)
    kwargs['runtime_library_dirs'] = runtime_library_dirs

    kwargs['language'] = 'c++'
    kwargs['extra_compile_args'] = [
        '-std=c++14', '-fopenmp', '-D_GLIBCXX_USE_CXX11_ABI=0'
    ]
    return setuptools.Extension(name, sources, *args, **kwargs)


def cpp_extension_with_tf(name, sources, *args, **kwargs):
    '''
    Extension for Tensorflow
    '''
    import tensorflow as tf
    ext = CppExtension(name, sources, *args, **kwargs)
    ext.include_dirs.append(tf.sysconfig.get_include())
    ext.library_dirs.append(tf.sysconfig.get_lib())
    ext.libraries.append(':libtensorflow_framework.so.2')
    ext.runtime_library_dirs.append(tf.sysconfig.get_lib())
    return ext


def cpp_extension_with_pytorch(name, sources, *args, **kwargs):
    from torch.utils.cpp_extension import CppExtension as _CppExtension
    ext = CppExtension(name, sources, *args, **kwargs)
    _ext = _CppExtension(name, sources, *args, **kwargs)
    _ext.include_dirs.extend(ext.include_dirs)
    _ext.library_dirs.extend(ext.library_dirs)
    _ext.libraries.extend(ext.libraries)
    _ext.runtime_library_dirs.extend(ext.runtime_library_dirs)
    _ext.runtime_library_dirs.extend(_ext.library_dirs)
    _ext.extra_compile_args.extend(ext.extra_compile_args)
    _ext.extra_link_args.extend(ext.extra_link_args)
    extra_compile_args = [
        '-DTORCH_API_INCLUDE_EXTENSION_H',
        '-DTORCH_EXTENSION_NAME=' + name.split('.')[-1]
    ]
    _ext.extra_compile_args.extend(extra_compile_args)
    return _ext


def get_max_jobs(parallel):
    import multiprocessing
    max_jobs = os.getenv('MAX_JOBS', parallel)
    return str(max_jobs or multiprocessing.cpu_count())


def parallel_compile_extension():
    # NPY_NUM_BUILD_JOBS
    from numpy.distutils.ccompiler import CCompiler_compile
    import distutils.ccompiler
    distutils.ccompiler.CCompiler.compile = CCompiler_compile
