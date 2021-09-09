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

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "common/types.h"
#include "proto/types.pb.h"

#pragma pack(push, 1)

namespace galileo {
namespace common {

struct EntityRequest {
  size_t Capacity() const { return types_.Capacity() + counts_.Capacity(); }
  ArraySpec<uint8_t> types_;
  ArraySpec<uint32_t> counts_;
};

struct VertexReply {
  size_t Capacity() const {
    size_t total_size = sizeof(size_t);
    for (size_t i = 0; i < ids_.size(); ++i) {
      total_size += ids_[i].Capacity();
    }
    return total_size;
  }
  std::vector<ArraySpec<galileo::common::VertexID>> ids_;
};

struct EdgeReply {
  size_t Capacity() const {
    size_t total_size = sizeof(size_t);
    for (size_t i = 0; i < ids_.size(); ++i) {
      total_size += ids_[i].Capacity();
    }
    return total_size;
  }
  std::vector<ArraySpec<galileo::common::EdgeID>> ids_;
};

struct NeighborRequest {
  size_t Capacity() const {
    return ids_.Capacity() + edge_types_.Capacity() + sizeof(cnt) +
           sizeof(need_weight_);
  }
  ArraySpec<galileo::common::VertexID> ids_;
  ArraySpec<uint8_t> edge_types_;
  uint32_t cnt;
  bool need_weight_;
};

struct NeighborReplyWithWeight {
  size_t Capacity() const {
    size_t neighbors_size = sizeof(size_t);
    for (auto &neighbor : neighbors_) {
      neighbors_size += neighbor.Capacity();
    }
    return neighbors_size;
  }
  std::vector<ArraySpec<IDWeight>> neighbors_;
};

struct NeighborReplyWithoutWeight {
  size_t Capacity() const {
    size_t neighbors_size = sizeof(size_t);
    for (auto &neighbor : neighbors_) {
      neighbors_size += neighbor.Capacity();
    }
    return neighbors_size;
  }
  std::vector<ArraySpec<VertexID>> neighbors_;
};

struct VertexFeatureRequest {
  size_t Capacity() const {
    size_t fname_size = sizeof(size_t);
    for (auto &name : features_) {
      fname_size += name.Capacity();
    }
    return ids_.Capacity() + fname_size + max_dims_.Capacity();
  }
  ArraySpec<galileo::common::VertexID> ids_;
  std::vector<ArraySpec<char>> features_;
  ArraySpec<uint32_t> max_dims_;
};

struct EdgeFeatureRequest {
  size_t Capacity() const {
    size_t fname_size = sizeof(size_t);
    for (auto &name : features_) {
      fname_size += name.Capacity();
    }
    return ids_.Capacity() + fname_size + max_dims_.Capacity();
  }
  ArraySpec<galileo::common::EdgeID> ids_;
  std::vector<ArraySpec<char>> features_;
  ArraySpec<uint32_t> max_dims_;
};

struct FeatureReply {
  size_t Capacity() const {
    size_t feature_size = sizeof(size_t);
    for (auto &feature : features_) {
      feature_size += sizeof(size_t);
      for (auto &f : feature) {
        feature_size += f.Capacity();
      }
    }
    return features_type_.Capacity() + feature_size;
  }
  std::vector<std::vector<ArraySpec<char>>> features_;
  ArraySpec<galileo::proto::DataType> features_type_;
};

}  // namespace common
}  // namespace galileo

#pragma pack(pop)
