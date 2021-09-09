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

#include <cstdint>
#include "client/dgraph_type.h"

namespace galileo {
namespace client {

class DGraphCutter {
 public:
  void Reset(uint32_t partition_num, uint32_t shard_num);
  uint32_t IDCut(VertexID id) const;

 private:
  uint32_t partition_num_;
  uint32_t shard_num_;
};

}  // namespace client
}  // namespace galileo
