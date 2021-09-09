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
'''
global backend
'''
from galileo.platform.export import export
from galileo.platform.default_values import DefaultValues

__backend = DefaultValues.BACKEND


@export('galileo.unify')
def get_backend():
    return __backend.lower()


@export('galileo.unify')
def is_tf():
    return get_backend() == 'tf'


@export('galileo.unify')
def is_pytorch():
    return get_backend() == 'pytorch'


@export('galileo.unify')
def set_backend(backend):
    backend = backend.lower()
    assert backend in ['tf', 'pytorch']
    global __backend
    __backend = backend


@export('galileo.unify')
def use_tf():
    set_backend('tf')


@export('galileo.unify')
def use_pytorch():
    set_backend('pytorch')
