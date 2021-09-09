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

add_executable(packer_test packer_test.cc)
target_link_libraries(packer_test common gtest gtest_main)
add_test(NAME packer_test COMMAND packer_test)

add_executable(discoverer_test discoverer_test.cc)
target_link_libraries(discoverer_test
    ${CMAKE_THREAD_LIBS_INIT}
    discoverer gtest gtest_main)
add_test(NAME discoverer_test
    COMMAND discoverer_test)

