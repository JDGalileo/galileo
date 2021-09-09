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

#include <assert.h>
#include <string>

namespace galileo {
namespace service {

struct GraphConfig {
  uint32_t shard_index;
  uint32_t shard_count;
  int thread_num;
  std::string hdfs_addr;
  uint16_t hdfs_port;
  std::string schema_path;
  std::string data_path;
  std::string zk_addr;
  std::string zk_path;
  bool IsLocal() const {
    return hdfs_addr.find("hdfs://") == std::string::npos;
  }
};

}  // namespace service
}  // namespace galileo
