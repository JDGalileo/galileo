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

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

namespace galileo {
namespace common {

using VertexID = int64_t;
using VertexType = uint8_t;
using EdgeType = uint8_t;
using WeightType = float;

template <typename Type>
struct ArraySpec {
  ArraySpec() : data(nullptr), cnt(0) {}
  ArraySpec(const Type *val, size_t cnt) : data(val), cnt(cnt) {}
  ~ArraySpec(){}
  size_t Capacity() const { return sizeof(cnt) + sizeof(Type) * cnt; }

  bool IsEmpty() const { return 0 == cnt; }

  std::string Serialize(int limit = 50) const {
    int tmp_len = cnt > limit ? limit : cnt;
    if (std::is_same<Type, char>::value) {
      return std::string((const char *)data, tmp_len);
    }
    std::string str("[");
    for (int i = 0; i < tmp_len; ++i) {
      str += std::to_string(data[i]);
    }
    str += "]";
    return str;
  }
  const Type *data;
  size_t cnt;
};

struct EdgeArraySpec {
  EdgeArraySpec(const ArraySpec<VertexID> &src, const ArraySpec<VertexID> &dst,
                const ArraySpec<uint8_t> &type)
      : srcs(src), dsts(dst), types(type) {}
  size_t Capacity() const {
    return srcs.Capacity() + dsts.Capacity() + types.Capacity();
  }
  ArraySpec<VertexID> srcs;
  ArraySpec<VertexID> dsts;
  ArraySpec<uint8_t> types;
};

#pragma pack(push, 1)

struct NeighborInfo {
  VertexID vertex_id;
  WeightType weight;
  EdgeType edge_type;
};

struct EdgeID {
  EdgeType edge_type;
  VertexID src_id;
  VertexID dst_id;
  
  std::string DebugStr() {
    std::ostringstream oss;
    oss<<"EdgeID[type="<<(int)edge_type<<",src="<<src_id
        <<",dst="<<dst_id<<"]";
    return oss.str();
  }
};

struct EdgeIDPtr {
  EdgeID *ptr = nullptr;
  bool operator==(const EdgeIDPtr &o) const {
    if (nullptr == ptr) {
      return false;
    }
    return ptr->src_id == o.ptr->src_id && ptr->dst_id == o.ptr->dst_id &&
           ptr->edge_type == o.ptr->edge_type;
  }

  std::string DebugStr() {
    return ptr->DebugStr();
  }
};

struct EdgeIDPtrHash {
  size_t operator()(const EdgeIDPtr &id) const {
    if (nullptr == id.ptr) {
      return 0;
    }
    size_t hash = static_cast<size_t>(id.ptr->edge_type);
    EdgeType *remainder = reinterpret_cast<EdgeType *>(id.ptr);
    int *operand = reinterpret_cast<int *>(remainder + 1);
    for (size_t idx = 0; idx < 4; ++idx) {
      hash *= 37;
      hash += static_cast<size_t>(operand[idx]);
    }
    return hash;
  }
};

struct IDWeight {
  VertexID id_;
  float weight_;
};

#pragma pack(pop)

struct ShardMeta {
  uint32_t num_shards;
  uint32_t num_partitions;
  size_t vertex_size;                    // total vertex size
  size_t edge_size;                      // total edge size
  std::vector<float> vertex_weight_sum;  // by vertex type
  std::vector<float> edge_weight_sum;    // by edge type
};

// pass ip:port to ShardCallback
using ShardCallback = std::function<void(const std::string &)>;

struct ShardCallbacks {
  ShardCallback on_shard_online;
  ShardCallback on_shard_offline;
};

enum OperatorType {
  SAMPLE_VERTEX,
  SAMPLE_EDGE,

  SAMPLE_NEIGHBOR,
  GET_TOPK_NEIGHBOR,

  GET_VERTEX_FEATURE,
  GET_EDGE_FEATURE,
  GET_NEIGHBOR,
};

enum Consts : int {
  MAX_THRAED_NUM = 64,
  MAX_PATH_LEN = 1024,
  MAX_8BIT_NUM = 255,
  MEMORY_SIZE = 4 * 1024 * 1024,
  WRITE_BUFFER_SIZE = 64 * 1024 * 1024,
};

enum InvalidType : uint8_t { INVALID_TYPE = 0xff };

}  // namespace common
}  // namespace galileo
