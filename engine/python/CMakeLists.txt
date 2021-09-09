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

pybind11_add_module(py_client client.cc)
target_link_libraries(py_client PUBLIC client)

pybind11_add_module(py_service service.cc)
target_link_libraries(py_service PUBLIC service)

pybind11_add_module(py_convertor convertor.cc)
target_link_libraries(py_convertor PUBLIC convertor)
