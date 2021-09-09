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


def add_to_path(module, submodule):
    r'''
    Make sure directory containing top level submodules is in
    the __path__ so that "from galileo.foo import bar" works.
    '''
    path = os.path.abspath(os.path.dirname(submodule.__file__))
    if not hasattr(module, '__path__'):
        module.__path__ = [path]
    elif path not in module.__path__:
        module.__path__.append(path)


this = sys.modules[__name__]

from . import framework, platform

add_to_path(this, framework)
add_to_path(this, platform)

del add_to_path, framework, platform, this, os, sys
