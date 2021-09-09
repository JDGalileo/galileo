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

#include <pthread.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/message.h"
#include "common/packer.h"
#include "common/sampler.h"
#include "common/types.h"
#include "service/edge.h"
#include "service/entity_pool_manager.h"

namespace galileo {
namespace service {

template <typename T>
using SimpleSampler = galileo::common::SimpleSampler<T>;

template <typename T, typename Y>
using WeightedSampler = galileo::common::WeightedSampler<T, Y>;

class Graph;

struct VertexWeightComparision {
  bool operator()(const galileo::common::IDWeight& a,
                  const galileo::common::IDWeight& b) {
    return a.weight_ > b.weight_;
  }
};

class Vertex {
 public:
  Vertex(uint8_t vtype);
  ~Vertex();
  galileo::common::VertexID GetId() const;
  inline uint8_t GetType() const;
  float GetWeight() const;
  const char* GetFeature(const std::string& attr_name) const;

  bool DeSerialize(uint8_t type, const char* s, size_t size);
  inline bool DeSerialize(uint8_t type, const std::string& data);

  bool AddEdge(Edge* edge);

  void SampleNeighbor(const galileo::common::NeighborRequest& neighbor_request,
                      galileo::common::Packer* packer);
  void GetTopKNeighbor(const galileo::common::NeighborRequest& neighbor_request,
                       galileo::common::Packer* packer);
  void GetFullNeighbor(const galileo::common::NeighborRequest& neighbor_request,
                       galileo::common::Packer* packer);

  bool BuildSubEdgeSampler();

 private:
  bool _HadEdge();
  bool _IsDirectStore(const std::string& field_name) const;
  bool _IsDirectStore(size_t var_index) const;

  Vertex(const Vertex&) = delete;
  Vertex& operator=(const Vertex&) = delete;

 private:
  SimpleSampler<uint8_t> edge_type_samplers_;
  std::unordered_map<uint8_t, std::vector<Edge*>*> out_edges_;
  std::vector<WeightedSampler<Edge, galileo::common::EdgeIDPtr>> edge_samplers_;
  std::vector<uint32_t> edge_counts_;
  char* raw_data_;
};

uint8_t Vertex::GetType() const {
  return *reinterpret_cast<const uint8_t*>(raw_data_);
}

bool Vertex::DeSerialize(uint8_t type, const std::string& data) {
  return DeSerialize(type, data.c_str(), data.size());
}

}  // namespace service
}  // namespace galileo
