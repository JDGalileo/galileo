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

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/protocol.h>
#include <condition_variable>
#include <memory>

#include "client/dgraph_type.h"
#include "common/types.h"
#include "discovery/discoverer.h"
#include "discovery/serialize.h"
#include "proto/rpc.pb.h"

namespace galileo {
namespace rpc {

void HandleResponse(brpc::Controller* cntl,
                    galileo::proto::QueryResponse* response,
                    std::function<void(bool)> callback);

class Client {
 public:
  Client();
  ~Client();

 public:
  void Query(const galileo::proto::QueryRequest& request,
             galileo::proto::QueryResponse* response,
             std::function<void(bool)> callback);

 public:
  bool Init(uint32_t shard_id,
            std::shared_ptr<galileo::discovery::Discoverer> discoverer,
            const galileo::client::DGraphConfig& config);
  brpc::Channel* GetChannel();
  void AddChannel(const std::string& host_port);
  void RemoveChannel(const std::string& host_port);

 private:
  uint32_t shard_index_;
  std::shared_ptr<galileo::discovery::Discoverer> discoverer_;
  brpc::Channel* channel_;
  galileo::common::ShardCallbacks shard_callbacks_;

  std::mutex mt_;
  std::condition_variable cv_;

  int64_t timeout_ms_;
};

}  // namespace rpc
}  // namespace galileo
