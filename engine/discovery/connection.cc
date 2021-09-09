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

#include <thread>

#include "discovery/connection.h"
#include "discovery/consts.h"

namespace galileo {
namespace discovery {

using namespace std::chrono_literals;  // NOLINT

Connection::Connection(const std::string &address)
    : address_(address), handle_(nullptr) {}

Connection::~Connection() { Close(); }

zhandle_t *Connection::GetHandle() { return handle_; }

void Connection::_WaitConnect() {
  std::unique_lock<std::mutex> lk(mt_);
  // wait for 10minutes
  bool res = cv_.wait_for(lk, 10min, [&] { return handle_ != nullptr; });
  if (!res) {
    std::string error = "Wait for zookeeper connected timeout after 10min";
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}

void Connection::Connected(zhandle_t *handle) {
  std::unique_lock<std::mutex> lk(mt_);
  handle_ = handle;
  cv_.notify_all();
}

void Connection::Close() {
  std::unique_lock<std::mutex> lk(mt_);
  zookeeper_close(handle_);
  handle_ = nullptr;
}

}  // namespace discovery
}  // namespace galileo
