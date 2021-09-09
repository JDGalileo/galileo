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

#include "client/rpc_client.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace bthread {
DECLARE_int32(bthread_concurrency);
}

namespace galileo {
namespace rpc {

void HandleResponse(brpc::Controller* cntl,
                    galileo::proto::QueryResponse* response,
                    std::function<void(bool)> callback) {
  // std::unique_ptr makes sure cntl will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);

  if (nullptr == response) {
    LOG(ERROR) << "Rpc response is nullptr.";
    callback(false);
  }
  if (cntl->Failed()) {
    LOG(ERROR) << "Fail to send QueryRequest, " << cntl->ErrorText();
    callback(false);
    return;
  }

  // set status true, then callback can deal response
  callback(true);
}

Client::Client()
    : shard_index_(0),
      channel_(nullptr),
      shard_callbacks_{
          std::bind(&Client::AddChannel, this, std::placeholders::_1),
          std::bind(&Client::RemoveChannel, this, std::placeholders::_1)} {}

Client::~Client() {
  if (discoverer_) {
    discoverer_->UnsetShardCallbacks(shard_index_, &shard_callbacks_);
  }
  if (channel_) {
    delete channel_;
    channel_ = nullptr;
  }
}

bool Client::Init(uint32_t shard_id,
                  std::shared_ptr<galileo::discovery::Discoverer> discoverer,
                  const galileo::client::DGraphConfig& config) {
  timeout_ms_ = config.rpc_timeout_ms;
  brpc::FLAGS_max_body_size = config.rpc_body_size;
  bthread::FLAGS_bthread_concurrency = config.rpc_bthread_concurrency;
  LOG(INFO) << " Brpc config info:"
            << " timeout_ms:" << timeout_ms_
            << ", max_body_size:" << config.rpc_body_size
            << ", bthread_concurrency:" << config.rpc_bthread_concurrency;
  shard_index_ = shard_id;
  if (nullptr != discoverer) {
    discoverer->SetShardCallbacks(shard_index_, &shard_callbacks_);
    discoverer_ = discoverer;
    return true;
  } else {
    return false;
  }
}

brpc::Channel* Client::GetChannel() {
  std::unique_lock<std::mutex> lock(mt_);
  cv_.wait(lock, [this] { return channel_ != nullptr; });
  return channel_;
}

void Client::AddChannel(const std::string& host_port) {
  LOG(INFO) << "add channel:" << host_port;
  std::lock_guard<std::mutex> lock(mt_);
  brpc::Channel* channel = new brpc::Channel();
  // Initialize the channel, NULL means using default options.
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  options.timeout_ms = 100 /*milliseconds*/;
  options.max_retry = 3;

  if (channel->Init(host_port.c_str(), "", &options) == 0) {
    if (channel_ != nullptr) {
      delete channel_;
      channel_ = nullptr;
    }
    channel_ = channel;
    cv_.notify_all();
  } else {
    delete channel;
    channel_ = nullptr;
  }
}

void Client::RemoveChannel(const std::string& host_port) {
  std::lock_guard<std::mutex> lock(mt_);
  if (channel_ != nullptr) {
    delete channel_;
    channel_ = nullptr;
  }
  LOG(INFO) << "remove channel:" << host_port;
}

void Client::Query(const galileo::proto::QueryRequest& request,
                   galileo::proto::QueryResponse* response,
                   std::function<void(bool)> callback) {
  galileo::proto::GraphQueryService_Stub stub(this->GetChannel());

  brpc::Controller* cntl = new brpc::Controller();

  cntl->set_timeout_ms(timeout_ms_);
  cntl->ignore_eovercrowded();

  google::protobuf::Closure* done =
      brpc::NewCallback(&HandleResponse, cntl, response, callback);

  stub.Query(cntl, &request, response, done);
}

}  // namespace rpc
}  // namespace galileo
