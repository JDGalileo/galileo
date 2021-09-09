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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common/sampler.h"
#include "common/types.h"
#include "service/entity_pool_manager.h"

namespace galileo {
namespace service {

class Edge {
 public:
  Edge(uint8_t etype);
  ~Edge();

  inline uint8_t GetType() const;
  inline galileo::common::EdgeIDPtr GetId() const;
  galileo::common::VertexID GetSrcVertex() const;
  galileo::common::VertexID GetDstVertex() const;
  float GetWeight() const;
  float GetFixFeatureWithOffset(size_t offset,
                                const std::string& feature_type) const;
  const char* GetFeature(const std::string& attr_name) const;

  bool DeSerialize(uint8_t type, const char* s, size_t size);
  inline bool DeSerialize(uint8_t type, const std::string& data);

  inline std::string DebugStr();

 private:
  bool _IsDirectStore(const std::string& field_name) const;
  bool _IsDirectStore(size_t var_index) const;

  Edge(const Edge&) = delete;
  Edge& operator=(const Edge&) = delete;

 private:
  char* raw_data_;
};

uint8_t Edge::GetType() const {
  return *reinterpret_cast<const uint8_t*>(raw_data_);
}

galileo::common::EdgeIDPtr Edge::GetId() const {
  return {(galileo::common::EdgeID*)raw_data_};
}

bool Edge::DeSerialize(uint8_t type, const std::string& data) {
  return DeSerialize(type, data.c_str(), data.size());
}

std::string Edge::DebugStr() {
  return GetId().DebugStr();
}

}  // namespace service
}  // namespace galileo
