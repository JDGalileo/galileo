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

#include "discovery/serialize.h"
#include "utils/string_util.h"

#include "glog/logging.h"

#include <algorithm>

namespace galileo {
namespace discovery {

bool ShardPath::_Check() const {
  if (shard_address.empty()) {
    LOG(WARNING) << "shard address must be not empty";
    return false;
  }
  return true;
}

bool ShardPath::Serialize(std::string *out) const {
  if (!this->_Check()) {
    return false;
  }
  *out = utils::join_string({std::to_string(shard_index), shard_address}, "#");
  return true;
}

bool ShardPath::Deserialize(const std::string &value) {
  std::vector<std::string> res;
  if (2 > utils::split_string(value, '#', &res)) {
    LOG(WARNING) << " Faild deserialize shard path";
    return false;
  }
  try {
    shard_index = static_cast<decltype(shard_index)>(std::stoul(res[0]));
  } catch (std::logic_error &e) {
    LOG(WARNING) << " Faild deserialize shard path shard_index convert error";
    return false;
  }
  shard_address = std::move(res[1]);
  return this->_Check();
}

bool ShardPath::operator==(const ShardPath &other) const {
  return shard_index == other.shard_index &&
         shard_address == other.shard_address;
}

namespace shard_meta {

bool Check(const ShardMeta &sm) {
  if (sm.num_shards < 1) {
    LOG(WARNING) << " Num shards must be > 0";
    return false;
  }
  if (sm.num_partitions < 1) {
    LOG(WARNING) << " Num partitions must be > 0";
    return false;
  }
  if (sm.vertex_weight_sum.empty()) {
    LOG(WARNING) << " Vertex weight sum must be not empty";
    return false;
  }
  if (sm.edge_weight_sum.empty()) {
    LOG(WARNING) << " Edge weight sum must be not empty";
    return false;
  }
  return true;
}

bool Serialize(const ShardMeta &sm, std::string *out) {
  if (!shard_meta::Check(sm)) {
    return false;
  }
  std::vector<std::string> vertex_weight_sum_str(sm.vertex_weight_sum.size());
  std::transform(sm.vertex_weight_sum.begin(), sm.vertex_weight_sum.end(),
                 vertex_weight_sum_str.begin(),
                 [](float value) { return std::to_string(value); });
  std::vector<std::string> edge_weight_sum_str(sm.edge_weight_sum.size());
  std::transform(sm.edge_weight_sum.begin(), sm.edge_weight_sum.end(),
                 edge_weight_sum_str.begin(),
                 [](float value) { return std::to_string(value); });
  *out = utils::join_string(
      {std::to_string(sm.num_shards), std::to_string(sm.num_partitions),
       std::to_string(sm.vertex_size), std::to_string(sm.edge_size),
       utils::join_string(vertex_weight_sum_str, ","),
       utils::join_string(edge_weight_sum_str, ",")},
      "|");
  return true;
}

bool Deserialize(const std::string &value, ShardMeta *out) {
  std::vector<std::string> meta;
  if (6 > utils::split_string(value, '|', &meta)) {
    LOG(WARNING) << " Faild deserialize shard meta";
    return false;
  }
  try {
    out->num_shards =
        static_cast<decltype(out->num_shards)>(std::stoul(meta[0]));
  } catch (std::logic_error &e) {
    LOG(WARNING) << " Faild deserialize shard meta num_shards convert error";
    return false;
  }
  try {
    out->num_partitions =
        static_cast<decltype(out->num_partitions)>(std::stoul(meta[1]));
  } catch (std::logic_error &e) {
    LOG(WARNING)
        << " Faild deserialize shard meta num_partitions convert error";
    return false;
  }
  try {
    out->vertex_size = std::stoull(meta[2]);
  } catch (std::logic_error &e) {
    LOG(WARNING) << " Faild deserialize shard meta vertex_size convert error";
    return false;
  }
  try {
    out->edge_size = std::stoull(meta[3]);
  } catch (std::logic_error &e) {
    LOG(WARNING) << " Faild deserialize shard meta edge_size convert error";
    return false;
  }
  std::vector<std::string> weight_sum;
  auto size = utils::split_string(meta[4], ',', &weight_sum);
  out->vertex_weight_sum.resize(size);
  for (size_t i = 0; i < size; ++i) {
    try {
      out->vertex_weight_sum[i] = std::stof(weight_sum[i]);
    } catch (std::logic_error &e) {
      LOG(WARNING)
          << " Faild deserialize shard meta vertex_weight_sum convert error";
      return false;
    }
  }
  size = utils::split_string(meta[5], ',', &weight_sum);
  out->edge_weight_sum.resize(size);
  for (size_t i = 0; i < size; ++i) {
    try {
      out->edge_weight_sum[i] = std::stof(weight_sum[i]);
    } catch (std::logic_error &e) {
      LOG(WARNING)
          << " Faild deserialize shard meta edge_weight_sum convert error";
      return false;
    }
  }
  return shard_meta::Check(*out);
}

}  // namespace shard_meta

bool ShardCacheMap::IsShardAvailable() {
  if (cache.empty()) {
    return false;
  }
  auto &c = cache.begin()->second;
  return !c.addresses.empty() && c.meta;
}

bool ShardCacheMap::IsShardAvailable(uint32_t shard_index) {
  if (cache.count(shard_index) == 0) {
    return false;
  }
  auto &c = cache.at(shard_index);
  return !c.addresses.empty() && c.meta;
}

}  // namespace discovery
}  // namespace galileo
