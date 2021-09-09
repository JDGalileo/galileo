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
#include <thread>
#include <unordered_map>

#include "service/config.h"

namespace galileo {
namespace service {

class Service {
 public:
  Service();
  ~Service();

 public:
  void Start(const GraphConfig&);
  void Start(int, const GraphConfig&);

 private:
  std::string _GetIpPort(int) const;
  int32_t _GetThreadNum(int32_t);
};

void StartService(const GraphConfig& config, int port, bool daemon);

}  // namespace service
}  // namespace galileo
