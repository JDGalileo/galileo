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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/types.h"

namespace galileo {
namespace discovery {

struct ShardPath {
  uint32_t shard_index;
  std::string shard_address;  // ip:port

  // shard_index#shard_address
  bool Serialize(std::string *) const;
  bool Deserialize(const std::string &);
  bool operator==(const ShardPath &other) const;

 private:
  bool _Check() const;
};

struct ShardPathHash {
  size_t operator()(const ShardPath &sp) const {
    std::string sp_str;
    sp.Serialize(&sp_str);
    return std::hash<std::string>()(sp_str);
  }
};

using ShardMeta = galileo::common::ShardMeta;
using ShardCallbacks = galileo::common::ShardCallbacks;

namespace shard_meta {
bool Serialize(const ShardMeta &, std::string *);
bool Deserialize(const std::string &, ShardMeta *);
bool Check(const ShardMeta &);
}  // namespace shard_meta

struct ShardCache {
  std::unordered_set<std::string> addresses;
  std::unordered_set<const ShardCallbacks *> callbacks;
  std::shared_ptr<ShardMeta> meta;
};

struct ShardCacheMap {
  std::unordered_map<uint32_t, ShardCache> cache;
  bool IsShardAvailable();
  bool IsShardAvailable(uint32_t);
};

using ShardMap = std::unordered_map<ShardPath, ShardMeta, ShardPathHash>;

}  // namespace discovery
}  // namespace galileo
