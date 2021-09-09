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

#include "discovery/register.h"
#include "utils/string_util.h"

namespace galileo {
namespace discovery {

Register::Register(const std::string &zk_addr, const std::string &zk_path)
    : connection_(zk_addr), path_(zk_path) {
  connection_.Connect(this, _ConnectionWatcher);
  _CreateRootNode();
}

void Register::_ConnectionWatcher(zhandle_t *handle, int, int state,
                                  const char *, void *context) {
  auto ctx = static_cast<Register *>(context);
  if (state == ZOO_CONNECTED_STATE) {
    ctx->connection_.Connected(handle);
  } else if (state == ZOO_EXPIRED_SESSION_STATE) {
    ctx->connection_.Reconnect(ctx, _ConnectionWatcher);
    for (const auto &s : ctx->shards_) {
      ctx->_CreateNode(s.first, s.second);
    }
  }
}

void Register::_CreateNode(const ShardPath &shard_path, const ShardMeta &meta,
                           uint32_t max_retry_times) {
  std::string sp_str;
  if (!shard_path.Serialize(&sp_str)) {
    LOG(WARNING) << " Skip error shard path: " << sp_str;
    return;
  }
  std::string sm_str;
  if (!shard_meta::Serialize(meta, &sm_str)) {
    LOG(WARNING) << " Skip error shard meta: " << sm_str;
    return;
  }
  _CreateNode(sp_str, sm_str, max_retry_times);
}

void Register::_CreateNode(const std::string &path, const std::string &data,
                           uint32_t max_retry_times) {
  if (max_retry_times < 1) {
    max_retry_times = 1;
  }
  std::string full_path = utils::join_string({path_, path}, "/");
  int res = 1;
  for (uint32_t t = 1; t <= max_retry_times; ++t) {
    res = zoo_create(connection_.GetHandle(), full_path.c_str(), data.c_str(),
                     static_cast<int>(data.size()), &ZOO_OPEN_ACL_UNSAFE,
                     ZOO_EPHEMERAL, nullptr, 0);
    if (res != ZOK) {
      LOG(WARNING)
          << " Failed to create child node at zookeeper, retrying ... #" +
                 std::to_string(t);
      std::chrono::seconds sp(5 * t);
      std::this_thread::sleep_for(sp);
    } else {
      break;
    }
  }
  if (res != ZOK) {
    std::string error =
        "Failed to create child node at zookeeper, "
        "and reached the max retry times " +
        std::to_string(max_retry_times);
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}

void Register::_DeleteNode(const ShardPath &shard_path,
                           uint32_t max_retry_times) {
  std::string sp_str;
  if (!shard_path.Serialize(&sp_str)) {
    LOG(WARNING) << " Skip error shard path: " << sp_str;
    return;
  }
  _DeleteNode(sp_str, max_retry_times);
}

void Register::_DeleteNode(const std::string &path, uint32_t max_retry_times) {
  if (max_retry_times < 1) {
    max_retry_times = 1;
  }
  std::string full_path = utils::join_string({path_, path}, "/");
  int res = 1;
  for (uint32_t t = 1; t <= max_retry_times; ++t) {
    res = zoo_delete(connection_.GetHandle(), full_path.c_str(), -1);
    if (res != ZOK && res != ZNONODE) {
      LOG(WARNING) << "Failed to delete node at zookeeper, retrying ... #" +
                          std::to_string(t);
      std::chrono::seconds sp(5 * t);
      std::this_thread::sleep_for(sp);
    } else {
      break;
    }
  }
  if (res != ZOK && res != ZNONODE) {
    std::string error =
        "Failed to delete node at zookeeper, "
        "and reached the max retry times " +
        std::to_string(max_retry_times);
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}

void Register::_CreateRootNode(uint32_t max_retry_times) {
  if (max_retry_times < 1) {
    max_retry_times = 1;
  }
  if (path_[0] != '/') {
    path_ = "/" + path_;
  }
  int res = 1;
  for (uint32_t t = 1; t <= max_retry_times; ++t) {
    res = zoo_create(connection_.GetHandle(), path_.c_str(), "",
                     0,  // empty data
                     &ZOO_OPEN_ACL_UNSAFE, ZOO_PERSISTENT, nullptr, 0);
    if (res != ZOK && res != ZNODEEXISTS) {
      LOG(WARNING)
          << "Failed to create root node at zookeeper, recreating ... #" +
                 std::to_string(t);
      std::chrono::seconds sp(5 * t);
      std::this_thread::sleep_for(sp);
    } else {
      break;
    }
  }
  if (res != ZOK && res != ZNODEEXISTS) {
    std::string error =
        "Failed to create root node at zookeeper, "
        "and reached the max retry times " +
        std::to_string(max_retry_times);
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}

void Register::AddShard(const ShardPath &shard_path, const ShardMeta &meta) {
  _CreateNode(shard_path, meta);
  {
    std::lock_guard<std::mutex> lock(mt_);
    shards_.emplace(shard_path, meta);
  }
}

void Register::RemoveShard(const ShardPath &shard_path) {
  _DeleteNode(shard_path);
  {
    std::lock_guard<std::mutex> lock(mt_);
    shards_.erase(shard_path);
  }
}

}  // namespace discovery
}  // namespace galileo
