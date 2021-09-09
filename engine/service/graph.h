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
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "common/message.h"
#include "common/sampler.h"
#include "common/types.h"
#include "proto/types.pb.h"
#include "service/config.h"
#include "service/edge.h"
#include "service/vertex.h"

namespace galileo {
namespace service {

class GraphLoader;

class Graph {
 public:
  Graph() : num_shards_(0), num_partitions_(0) {}
  ~Graph();

  Vertex* GetVertexByID(const galileo::common::VertexID& id) {
    if (vertex_map_.find(id) != vertex_map_.end()) {
      return vertex_map_[id];
    } else {
      return nullptr;
    }
  }

  Edge* GetEdgeByID(galileo::common::EdgeID& id) {
    galileo::common::EdgeIDPtr e_id{&id};
    if (edge_map_.find(e_id) != edge_map_.end()) {
      return edge_map_.at(e_id);
    } else {
      return nullptr;
    }
  }

  void SetPartitions(uint32_t partitions_) {
    num_partitions_ = partitions_;
  }

  bool AddVertexs(std::vector<Vertex*>& vec);

  bool AddEdges(std::vector<Edge*>& vec);

  bool Init(const galileo::service::GraphConfig& config);
  const galileo::common::ShardMeta QueryShardMeta();

  size_t GetVertexCount() { return vertex_map_.size(); }
  size_t GetEdgeCount() { return edge_map_.size(); }

  bool BuildSubEdgeSampler();
  bool BuildGlobalVertexSampler();
  bool BuildGlobalEdgeSampler();

  bool SampleVertex(const galileo::common::EntityRequest& entity_request,
                    galileo::common::Packer* packer);
  bool SampleEdge(const galileo::common::EntityRequest& entity_request,
                  galileo::common::Packer* packer);
  bool QueryNeighbors(galileo::common::OperatorType type,
                      const galileo::common::NeighborRequest& neighbor_request,
                      galileo::common::Packer* packer);
  bool GetEdgeFeature(
      const galileo::common::EdgeFeatureRequest& edge_feature_request,
      galileo::common::Packer* packer);
  bool GetVertexFeature(
      const galileo::common::VertexFeatureRequest& vertex_feature_request,
      galileo::common::Packer* packer);

 private:
  void _SampleVertex(uint8_t vertex_type, uint32_t count,
                     galileo::common::Packer* packer);
  void _SampleEdge(uint8_t edge_type, uint32_t count,
                   galileo::common::Packer* packer);

 protected:
  std::unordered_map<galileo::common::VertexID, Vertex*> vertex_map_;
  std::unordered_map<galileo::common::EdgeIDPtr, Edge*,
                     galileo::common::EdgeIDPtrHash>
      edge_map_;

  std::vector<WeightedSampler<Vertex, galileo::common::VertexID>>
      vertex_samplers_;
  std::vector<WeightedSampler<Edge, galileo::common::EdgeIDPtr>> edge_samplers_;

  std::vector<float> vertex_sum_weight_;
  std::vector<float> edge_sum_weight_;

  std::vector<uint32_t> vertex_type_counts_;
  std::vector<uint32_t> edge_type_counts_;

  SimpleSampler<uint8_t> vertex_type_samplers_;
  SimpleSampler<uint8_t> edge_type_samplers_;

  std::shared_ptr<GraphLoader> graph_loader_;
  uint32_t num_shards_;
  uint32_t num_partitions_;
};

galileo::proto::DataType transformDataTypeByStrName(const std::string& type);

}  // namespace service
}  // namespace galileo
