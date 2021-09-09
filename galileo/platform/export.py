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

import sys


class export(object):
    r'''
    \brief Export galileo APIs
    '''
    def __init__(self, *args, **kwargs):
        r'''
        \param *args modules, expected modules, dot delimited format.
            Default use base_module, e.g. galileo or galileo.tf
        \param **kwargs base_module Default is `galileo`.
        '''
        self._modules = args
        self._base_module = kwargs.get('base_module', 'galileo')
        if not self._modules:
            self._modules = [self._base_module]

    def __call__(self, func):
        r'''
        \brief export class or function
        for @export
        '''
        for module in self._modules:
            m = sys.modules[module]
            setattr(m, func.__name__, func)
        return func

    def var(self, name, value):
        r'''
        \brief export variable

        \param name variable name
        \param value variable value
        '''
        for module in self._modules:
            m = sys.modules[module]
            setattr(m, name, value)

    def submodule(self, export_name, src_name):
        r'''
        \brief export submodule

        \param export_name export module name
        \param src_name src module name
        '''
        sub = sys.modules[src_name]
        for module in self._modules:
            m = sys.modules[module]
            setattr(m, export_name, sub)
            sys.modules[f'{module}.{export_name}'] = sub
