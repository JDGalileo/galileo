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

# not use import because galileo is not ready
exec(open('galileo/platform/version.py').read(), globals())
exec(open('galileo/platform/setuptools_helper.py').read(), globals())

from setuptools import setup, find_packages

import sys

package_name = 'jdgalileo'
tf_require = 'tensorflow'
if 'gpu' in package_name:
    tf_require = 'tensorflow-gpu'

description = 'Galileo library for large scale graph training by JD'
keywords = [
    'graph-embedding'
    'graph-neural-networks',
    'tensorflow',
    'pytorch',
]
install_requires = [
    tf_require + '>=2.3.0',
    'torch>=1.6.0',
    'networkx==2.3',
    'attrs',
]
setup_requires = ['numpy']
tests_require = ['pytest']
packages = find_packages(exclude=('*.tests', ))
package_data = {
    'galileo': [
        'framework/pywrap/*.so*',
        'framework/libs/lib*.so*',
    ]
}
cmdclass = {
    'build': build,
    'develop': develop,
    'build_ext': build_ext,
}
ext_modules = [
    cpp_extension_with_tf('galileo.framework.pywrap.tf_ops', [
        'galileo/framework/tf/kernel/entity_ops.cc',
        'galileo/framework/tf/kernel/feature_ops.cc',
        'galileo/framework/tf/kernel/neighbor_ops.cc',
        'galileo/framework/tf/kernel/dataset_ops.cc',
        'galileo/framework/tf/kernel/sequence_ops.cc',
        'galileo/framework/tf/ops/entity.cc',
        'galileo/framework/tf/ops/features.cc',
        'galileo/framework/tf/ops/neighbors.cc',
        'galileo/framework/tf/ops/dataset.cc',
        'galileo/framework/tf/ops/sequence.cc',
    ]),
    cpp_extension_with_pytorch('galileo.framework.pywrap.pt_ops', [
        'galileo/framework/pytorch/kernel/entity_ops.cc',
        'galileo/framework/pytorch/kernel/feature_ops.cc',
        'galileo/framework/pytorch/kernel/neighbor_ops.cc',
        'galileo/framework/pytorch/kernel/sequence_ops.cc',
        'galileo/framework/pytorch/ops/ops.cc',
    ]),
]
classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: GPU :: NVIDIA CUDA :: 10.1',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: C++',
]
entry_points = {
    'console_scripts': [
        'galileo_convertor = galileo.platform.tools.convertor:main',
        'galileo_service = galileo.platform.tools.start_service:main',
    ],
}

#parallel_compile_extension()

setup(name=package_name,
      version=__version__,
      description='Galileo',
      long_description=description,
      author='Galileo Authors',
      author_email='galileo_opensource@jd.com',
      license="Apache License 2.0",
      keywords=keywords,
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      packages=packages,
      package_data=package_data,
      python_requires='>=3.8',
      entry_points=entry_points,
      classifiers=classifiers)
