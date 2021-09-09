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

#include <memory>
#include <string>
#include <vector>

#include <zookeeper/zookeeper.h>

#include "discovery/connection.h"
#include "discovery/serialize.h"

namespace galileo {
namespace discovery {

class Discoverer {
 public:
  explicit Discoverer(const std::string &, const std::string &);
  uint32_t GetShardsNum();
  uint32_t GetPartitionsNum();
  uint64_t GetVertexSize();
  uint64_t GetEdgeSize();
  void GetVertexWeightSum(uint32_t, std::vector<float> *);
  void GetEdgeWeightSum(uint32_t, std::vector<float> *);

  void SetShardCallbacks(uint32_t, const ShardCallbacks *);
  void UnsetShardCallbacks(uint32_t, const ShardCallbacks *);

 private:
  Discoverer(const Discoverer &) = delete;
  Discoverer &operator=(const Discoverer &) = delete;

  std::shared_ptr<ShardMeta> _WaitShardReady();  // wait any one of shards
  std::shared_ptr<ShardMeta> _WaitShardReady(uint32_t);
  void _WaitAllShardReady();
  void _Notify(const ShardPath &, std::shared_ptr<ShardMeta>);

  static void _ConnectionWatcher(zhandle_t *zh, int type, int state,
                                 const char *path, void *context);
  static void _RootWatcher(zhandle_t *zh, int type, int state, const char *path,
                           void *context);
  static void _ShardWatcher(zhandle_t *zh, int type, int state,
                            const char *path, void *context);
  static void _RootCompletion(int rc, const struct Stat *stat,
                              const void *data);
  static void _ShardPathCompletion(int rc, const struct String_vector *strings,
                                   const void *data);
  static void _ShardMetaCompletion(int rc, const char *value, int value_len,
                                   const struct Stat *stat, const void *data);

  void _CheckRootNode();
  void _CheckShardNode();

  void _ShardOnline(const ShardPath &);
  void _ShardOffline(const ShardPath &);

 private:
  Connection connection_;
  std::string path_;
  ShardCacheMap shards_;
  std::unordered_set<ShardPath, ShardPathHash> shard_paths_;

  std::mutex mt_;
  std::condition_variable cv_;
};

}  // namespace discovery
}  // namespace galileo
