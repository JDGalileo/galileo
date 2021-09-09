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

# Minimum CMake required
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(GalileoServing VERSION 0.0.1 LANGUAGES CXX)

enable_testing()

# Set C++17 as standard for the whole project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O2")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-fno-omit-frame-pointer)

# Debug Release RelWithDebInfo MinSizeRel RelWithAssert
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)

# Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
# for targets that link ${CMAKE_THREAD_LIBS_INIT}.
find_package (Threads)

set(LIBS_PREFIX /usr/local)
include_directories(SYSTEM
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_BINARY_DIR}
    ${LIBS_PREFIX}/include
)
link_directories(BEFORE
    ${LIBS_PREFIX}/lib
    ${LIBS_PREFIX}/lib64
)

# Add external module
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(quickjson)

find_package(pybind11 CONFIG)

add_compile_options(-Wall -Wextra -pedantic -Wshadow
    -Wfloat-equal
    -Werror
    -Wconversion)

add_subdirectory(utils)
add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(client)
add_subdirectory(discovery)
add_subdirectory(service)
add_subdirectory(convertor)
add_subdirectory(tests)
add_subdirectory(python)

