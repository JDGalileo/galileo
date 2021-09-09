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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <zookeeper/zookeeper.h>

#include "discovery/connection.h"
#include "discovery/serialize.h"

namespace galileo {
namespace discovery {

class Register {
 public:
  // Connect sync and create root node
  explicit Register(const std::string &, const std::string &);
  // register one shard
  void AddShard(const ShardPath &shard_path, const ShardMeta &meta);
  void RemoveShard(const ShardPath &shard_path);

 private:
  Register(const Register &) = delete;
  Register &operator=(const Register &) = delete;

  void _CreateNode(const ShardPath &shard_path, const ShardMeta &meta,
                   uint32_t max_retry_times = kMaxRetryTimes);
  void _CreateNode(const std::string &path, const std::string &data,
                   uint32_t max_retry_times = kMaxRetryTimes);
  void _CreateRootNode(uint32_t max_retry_times = kMaxRetryTimes);
  void _DeleteNode(const ShardPath &shard_path,
                   uint32_t max_retry_times = kMaxRetryTimes);
  void _DeleteNode(const std::string &path,
                   uint32_t max_retry_times = kMaxRetryTimes);

  static void _ConnectionWatcher(zhandle_t *zh, int type, int state,
                                 const char *path, void *context);

  Connection connection_;
  std::string path_;
  ShardMap shards_;
  std::mutex mt_;
};

}  // namespace discovery
}  // namespace galileo
