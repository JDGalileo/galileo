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
from distutils.sysconfig import get_config_var

if __file__ == 'setup.py':
    # when setup
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
else:
    project_root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

galileo_src_dir = os.path.join(project_root_dir, 'galileo')

engine_src_dir = os.path.join(project_root_dir, 'engine')

engine_build_dir = os.path.join(project_root_dir, 'build', 'engine')

engine_client_dir = os.path.join(engine_build_dir, 'client')

engine_proto_dir = os.path.join(engine_build_dir, 'proto')

engine_python_dir = os.path.join(engine_build_dir, 'python')

libs_dir = os.path.join(project_root_dir, 'galileo', 'framework', 'libs')

pywrap_dir = os.path.join(project_root_dir, 'galileo', 'framework', 'pywrap')


def get_tf_ops():
    suffix = get_config_var('EXT_SUFFIX')
    return os.path.join(pywrap_dir, 'tf_ops' + suffix)


def get_cpp_targets():
    return [
        os.path.join(engine_client_dir, 'libclient.so'),
        os.path.join(engine_proto_dir, 'libproto.so'),
    ]


def get_py_targets():
    suffix = get_config_var('EXT_SUFFIX')
    return [
        os.path.join(engine_python_dir, 'py_client' + suffix),
        os.path.join(engine_python_dir, 'py_service' + suffix),
        os.path.join(engine_python_dir, 'py_convertor' + suffix),
    ]


def get_all_targets():
    return get_cpp_targets() + get_py_targets()


def is_targets_exists():
    return all([os.path.isfile(f) for f in get_all_targets()])
