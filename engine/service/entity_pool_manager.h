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
#include <vector>

#include "common/schema.h"
#include "common/singleton.h"
#include "utils/memory_pool.h"

namespace galileo {
namespace service {

using MemoryPool = galileo::utils::MemoryPool;
using Schema = galileo::schema::Schema;

class EntityPoolManager {
  friend class galileo::common::Singleton<EntityPoolManager>;

 private:
  EntityPoolManager() {}

 public:
  void Init(
      size_t vBlockSize =
          100000 /*vertexes in chunks of this size. default value = 100000*/,
      size_t eBlockSize =
          100000 /*edges in chunks of this size. default value = 100000*/) {
    Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
    for (uint8_t vtype = 0; vtype < schema->GetVTypeNum(); vtype++) {
      std::shared_ptr<MemoryPool> new_pool(
          new MemoryPool(vBlockSize, schema->GetVRecordSize(vtype)));
      vertexes_pool_.emplace_back(new_pool);
    }
    for (uint8_t etype = 0; etype < schema->GetETypeNum(); etype++) {
      std::shared_ptr<MemoryPool> new_pool(
          new MemoryPool(eBlockSize, schema->GetERecordSize(etype)));
      edges_pool_.emplace_back(new_pool);
    }
  }

  ~EntityPoolManager() {
    vertexes_pool_.clear();
    edges_pool_.clear();
  }

  std::shared_ptr<MemoryPool> GetVMemoryPool(const uint8_t vtype) noexcept {
    return vertexes_pool_.at(vtype);
  }

  std::shared_ptr<MemoryPool> GetEMemoryPool(const uint8_t vtype) noexcept {
    return edges_pool_.at(vtype);
  }

 private:
  // vertexes memory pool map
  std::vector<std::shared_ptr<MemoryPool>> vertexes_pool_;
  // edges memory pool map
  std::vector<std::shared_ptr<MemoryPool>> edges_pool_;
};

}  // namespace service
}  // namespace galileo
