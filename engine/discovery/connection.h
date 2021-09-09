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

#include <glog/logging.h>
#include <zookeeper/zookeeper.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "discovery/consts.h"

namespace galileo {
namespace discovery {

typedef void (*ConnectionWatcher)(zhandle_t *zh, int type, int state,
                                  const char *path, void *watcherCtx);

class Connection {
 public:
  explicit Connection(const std::string &address);
  ~Connection();

  zhandle_t *GetHandle();

  template <typename Context>
  void Connect(Context *, ConnectionWatcher watcher,
               uint32_t max_retry_times = kMaxRetryTimes);

  void Close();

  template <typename Context>
  void Reconnect(Context *context, ConnectionWatcher watcher,
                 uint32_t max_retry_times = kMaxRetryTimes) {
    Close();
    Connect(context, watcher, max_retry_times);
  }

  void Connected(zhandle_t *handle);

 private:
  void _WaitConnect();

  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;

  std::string address_;
  zhandle_t *handle_;
  std::mutex mt_;
  std::condition_variable cv_;
};

template <typename Context>
void Connection::Connect(Context *context, ConnectionWatcher watcher,
                         uint32_t max_retry_times) {
  zhandle_t *handle = nullptr;
  if (max_retry_times < 1) {
    max_retry_times = 1;
  }
  for (uint32_t t = 1; t <= max_retry_times; ++t) {
    handle = zookeeper_init(address_.c_str(), watcher, kSessionTimeoutMS,
                            /*clientid=*/nullptr, /*context=*/context, 0);
    if (handle == nullptr) {
      LOG(WARNING) << " Failed to create zookeeper client, "
                      "recreating ... #" +
                          std::to_string(t);
      std::chrono::seconds sp(5 * t);
      std::this_thread::sleep_for(sp);
    } else {
      break;
    }
  }
  if (handle == nullptr) {
    std::string error =
        " Failed to create zookeeper client, "
        "and reached the max retry times " +
        std::to_string(max_retry_times);
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
  this->_WaitConnect();
}

}  // namespace discovery
}  // namespace galileo
