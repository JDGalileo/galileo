// Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

#include "common/types.h"

namespace galileo {
namespace client {

using VertexID = int64_t;

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

enum ClientType : int32_t {
  CT_BOOL = 1,
  CT_UINT8,
  CT_INT8,
  CT_UINT16,
  CT_INT16,
  CT_UINT32,
  CT_INT32,
  CT_UINT64,
  CT_INT64,
  CT_FLOAT,
  CT_DOUBLE,
  CT_STRING,
  CT_INVALID_TYPE = -1,
};

class ITensorAlloc {
 public:
  virtual char* AllocListTensor(
      ClientType type, const std::initializer_list<long long>& dims) = 0;
  virtual char* AllocTypesTensor(long long count) { return nullptr; };

  virtual int GetTensorType(ClientType) { return -1; }
  virtual bool FillStringTensor(char* buffer, size_t idx,
                                const ArraySpec<char>& str) = 0;

  virtual bool FillStringTensor(char* buffer, size_t idx,
                                const std::string& str) = 0;
};

struct GraphMeta {
  size_t vertex_size;
  size_t edge_size;
};

struct DGraphConfig {
  std::string zk_addr = "";
  std::string zk_path = "";
  int64_t rpc_timeout_ms = -1;
  int32_t rpc_body_size = 2147483647;
  int32_t rpc_bthread_concurrency = 9;
};

}  // namespace client
}  // namespace galileo
