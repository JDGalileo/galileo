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

file(GLOB PROTO_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.proto")
set(PROTOC ${LIBS_PREFIX}/bin/protoc)

set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/proto)
set(PROTO_SRCS)

foreach(PF ${PROTO_FILES})
    get_filename_component(PN ${PF} NAME_WE)
    set(PROTO_SRC ${OUTPUT_DIR}/${PN}.pb.cc)
    set(PROTO_HDR ${OUTPUT_DIR}/${PN}.pb.h)
    add_custom_command(OUTPUT ${PROTO_SRC} ${PROTO_HDR}
        COMMAND ${PROTOC}
        ARGS --cpp_out=${OUTPUT_DIR}
        -I . ${PF}
        DEPENDS ${PF} ${PROTOC}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM)
    list(APPEND PROTO_SRCS ${PROTO_SRC})
endforeach()

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_C_VISIBILITY_PRESET default)

add_library(proto SHARED ${PROTO_SRCS})
target_link_libraries(proto protobuf)
