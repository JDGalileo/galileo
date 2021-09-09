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

#include "discovery/discoverer.h"
#include "glog/logging.h"
#include "utils/string_util.h"

namespace galileo {
namespace discovery {

Discoverer::Discoverer(const std::string &zk_addr, const std::string &zk_path)
    : connection_(zk_addr), path_(zk_path) {
  if (path_[0] != '/') {
    path_ = "/" + path_;
  }
  connection_.Connect(this, _ConnectionWatcher);
  _CheckRootNode();
}

uint32_t Discoverer::GetShardsNum() { return _WaitShardReady()->num_shards; }

uint32_t Discoverer::GetPartitionsNum() {
  return _WaitShardReady()->num_partitions;
}

size_t Discoverer::GetVertexSize() {
  _WaitAllShardReady();
  size_t size = 0;
  for (auto &shard : this->shards_.cache) {
    size += shard.second.meta->vertex_size;
  }
  return size;
}

size_t Discoverer::GetEdgeSize() {
  _WaitAllShardReady();
  size_t size = 0;
  for (auto &shard : this->shards_.cache) {
    size += shard.second.meta->edge_size;
  }
  return size;
}

void Discoverer::GetVertexWeightSum(uint32_t shard_index,
                                    std::vector<float> *out) {
  *out = _WaitShardReady(shard_index)->vertex_weight_sum;
}

void Discoverer::GetEdgeWeightSum(uint32_t shard_index,
                                  std::vector<float> *out) {
  *out = _WaitShardReady(shard_index)->edge_weight_sum;
}

void Discoverer::SetShardCallbacks(uint32_t shard_index,
                                   const ShardCallbacks *cb) {
  std::unique_lock<std::mutex> lock(mt_);
  auto &shard = shards_.cache[shard_index];
  if (!shard.callbacks.emplace(cb).second) {
    return;
  }
  lock.unlock();
  for (const auto &address : shard.addresses) {
    cb->on_shard_online(address);
  }
}

void Discoverer::UnsetShardCallbacks(uint32_t shard_index,
                                     const ShardCallbacks *cb) {
  std::lock_guard<std::mutex> lock(mt_);
  shards_.cache[shard_index].callbacks.erase(cb);
}

std::shared_ptr<ShardMeta> Discoverer::_WaitShardReady() {
  std::unique_lock<std::mutex> lock(mt_);
  cv_.wait(lock, [this] { return this->shards_.IsShardAvailable(); });
  auto &shard = this->shards_.cache.begin()->second;
  return shard.meta;
}

std::shared_ptr<ShardMeta> Discoverer::_WaitShardReady(uint32_t shard_index) {
  std::unique_lock<std::mutex> lock(mt_);
  cv_.wait(lock, [this, shard_index] {
    return this->shards_.IsShardAvailable(shard_index);
  });
  return this->shards_.cache.at(shard_index).meta;
}

void Discoverer::_WaitAllShardReady() {
  uint32_t num_shards = GetShardsNum();
  for (uint32_t i = 0; i < num_shards; ++i) {
    _WaitShardReady(i);
  }
}

void Discoverer::_ConnectionWatcher(zhandle_t *zh, int, int state, const char *,
                                    void *context) {
  auto ctx = static_cast<Discoverer *>(context);
  if (state == ZOO_CONNECTED_STATE) {
    ctx->connection_.Connected(zh);
  } else if (state == ZOO_EXPIRED_SESSION_STATE) {
    ctx->connection_.Reconnect(ctx, _ConnectionWatcher);
    ctx->_CheckRootNode();
  }
}

void Discoverer::_RootWatcher(zhandle_t *, int type, int, const char *,
                              void *context) {
  auto ctx = static_cast<Discoverer *>(context);
  if (ZOO_CREATED_EVENT == type) {
    ctx->_CheckShardNode();
  } else {
    LOG(ERROR) << " [zk] Watch root node type: " << type;
    // just ignore this error
  }
}

void Discoverer::_ShardWatcher(zhandle_t *, int type, int, const char *,
                               void *context) {
  auto ctx = static_cast<Discoverer *>(context);
  if (ZOO_CHILD_EVENT == type) {
    ctx->_CheckShardNode();
  } else if (ZOO_DELETED_EVENT == type) {
    ctx->_CheckRootNode();
  } else {
    LOG(ERROR) << " [zk] Watch child node type: " << type;
    // just ignore this error
  }
}

void Discoverer::_RootCompletion(int rc, const struct Stat *,
                                 const void *data) {
  if (ZOK == rc) {
    Discoverer *ctx = (Discoverer *)data;
    ctx->_CheckShardNode();
  } else {
    LOG(ERROR) << " [zk] Checking root node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

void Discoverer::_ShardPathCompletion(int rc,
                                      const struct String_vector *strings,
                                      const void *data) {
  if (ZOK == rc) {
    Discoverer *ctx = (Discoverer *)data;
    std::unordered_set<ShardPath, ShardPathHash> new_shard_paths;
    for (int32_t i = 0; i < strings->count; ++i) {
      ShardPath path;
      std::string p(strings->data[i]);
      if (path.Deserialize(p)) {
        new_shard_paths.insert(std::move(path));
      } else {
        LOG(WARNING) << " [zk] Skip invalid shard path: " << p;
      }
    }
    for (auto &p : new_shard_paths) {
      if (ctx->shard_paths_.count(p) == 0) {
        ctx->_ShardOnline(p);
      }
    }
    for (auto &p : ctx->shard_paths_) {
      if (new_shard_paths.count(p) == 0) {
        ctx->_ShardOffline(p);
      }
    }
    ctx->shard_paths_ = std::move(new_shard_paths);
  } else {
    LOG(ERROR) << " [zk] Checking child node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

using ShardPathDiscovery = std::pair<ShardPath, Discoverer *>;

void Discoverer::_ShardMetaCompletion(int rc, const char *value, int value_len,
                                      const struct Stat *, const void *data) {
  if (ZOK == rc) {
    std::unique_ptr<ShardPathDiscovery> spd((ShardPathDiscovery *)data);
    ShardPath sp = spd->first;
    Discoverer *ctx = spd->second;
    std::shared_ptr<ShardMeta> sm = std::make_shared<ShardMeta>();
    if (shard_meta::Deserialize(std::string(value, value_len), sm.get())) {
      ctx->_Notify(sp, sm);
    } else {
      std::string sp_str;
      sp.Serialize(&sp_str);
      LOG(WARNING) << " [zk] Skip invalid shard meta for: " << sp_str;
    }
  } else {
    LOG(ERROR) << " [zk] Get child node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

void Discoverer::_CheckRootNode() {
  int rc = zoo_awexists(connection_.GetHandle(), path_.c_str(), _RootWatcher,
                        this, _RootCompletion, this);
  if (ZOK != rc) {
    LOG(ERROR) << " [zk] Checking root node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

void Discoverer::_CheckShardNode() {
  int rc = zoo_awget_children(connection_.GetHandle(), path_.c_str(),
                              _ShardWatcher, this, _ShardPathCompletion, this);
  if (ZOK != rc) {
    LOG(ERROR) << " [zk] Checking child node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

void Discoverer::_ShardOnline(const ShardPath &p) {
  std::string shard_path;
  if (!p.Serialize(&shard_path)) {
    LOG(WARNING) << " Skip error shard path: " << shard_path;
    return;
  }
  LOG(INFO) << " Shard online: " << shard_path;
  int rc = zoo_aget(connection_.GetHandle(),
                    utils::join_string({path_, shard_path}, "/").c_str(), 0,
                    _ShardMetaCompletion, new ShardPathDiscovery(p, this));
  if (ZOK != rc) {
    LOG(ERROR) << " [zk] Get child node ERROR: " << zerror(rc);
    // just ignore this error
  }
}

void Discoverer::_ShardOffline(const ShardPath &p) {
  std::string shard_path;
  if (!p.Serialize(&shard_path)) {
    LOG(WARNING) << " Skip error shard path: " << shard_path;
    return;
  }
  LOG(INFO) << " Shard offline: " << shard_path;
  std::unique_lock<std::mutex> lock(mt_);
  auto &shard = shards_.cache.at(p.shard_index);
  shard.addresses.erase(p.shard_address);
  auto &callbacks = shard.callbacks;
  lock.unlock();
  for (auto &cb : callbacks) {
    cb->on_shard_offline(p.shard_address);
  }
}

void Discoverer::_Notify(const ShardPath &sp, std::shared_ptr<ShardMeta> sm) {
  std::unique_lock<std::mutex> lock(mt_);
  auto &shard = shards_.cache[sp.shard_index];
  shard.meta = sm;
  shard.addresses.emplace(sp.shard_address);
  cv_.notify_all();
  auto &callbacks = shard.callbacks;
  lock.unlock();
  for (auto &cb : callbacks) {
    cb->on_shard_online(sp.shard_address);
  }
}

}  // namespace discovery
}  // namespace galileo
