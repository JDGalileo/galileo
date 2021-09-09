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

#include "service/graph.h"
#include <fstream>
#include <sstream>

#include "common/macro.h"
#include "common/packer.h"
#include "common/schema.h"
#include "common/singleton.h"
#include "service/entity_pool_manager.h"
#include "service/graph_loader.h"
#include "utils/string_util.h"

namespace galileo {
namespace service {

using Schema = galileo::schema::Schema;

Graph::~Graph() {
  for (auto vertex : vertex_map_) {
    delete vertex.second;
  }
  vertex_map_.clear();
  for (auto edge : edge_map_) {
    delete edge.second;
  }
  edge_map_.clear();
}

bool Graph::Init(const galileo::service::GraphConfig& config) {
  vertex_map_.clear();
  edge_map_.clear();
  num_shards_ = config.shard_count;

  graph_loader_.reset(new GraphLoader(config));
  std::shared_ptr<galileo::utils::IFileReader> schema_file =
      graph_loader_->GetFileSystem()->OpenFileReader(
          config.schema_path.c_str());
  if (nullptr == schema_file.get()) {
    LOG(ERROR) << " Cant find the schema file";
    return false;
  }

  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();

  char buff[10 * 1024] = {0};
  schema_file->Read(buff, 10 * 1024, nullptr);
  if (unlikely(!schema->Build(buff))) {
    LOG(ERROR) << " Build schema fail!";
    return false;
  }

  galileo::common::Singleton<EntityPoolManager>::GetInstance()->Init();
  size_t vtype_num = static_cast<size_t>(schema->GetVTypeNum());
  size_t etype_num = static_cast<size_t>(schema->GetETypeNum());

  vertex_type_counts_.resize(vtype_num, 0);
  edge_type_counts_.resize(etype_num, 0);
  vertex_sum_weight_.resize(vtype_num, 0);
  edge_sum_weight_.resize(etype_num, 0);

  bool result = graph_loader_->LoadGraph(this, config);
  return result;
}

const galileo::common::ShardMeta Graph::QueryShardMeta() {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  galileo::common::ShardMeta shardMeta;
  shardMeta.num_shards = num_shards_;
  shardMeta.num_partitions = num_partitions_;
  shardMeta.vertex_size = vertex_map_.size();
  shardMeta.edge_size = edge_map_.size();
  size_t vtype_num = static_cast<size_t>(schema->GetVTypeNum());
  for (size_t i = 0; i < vtype_num; ++i) {
    shardMeta.vertex_weight_sum.push_back(vertex_sum_weight_[i]);
  }

  size_t etype_num = static_cast<size_t>(schema->GetETypeNum());
  for (size_t i = 0; i < etype_num; ++i) {
    shardMeta.edge_weight_sum.push_back(edge_sum_weight_[i]);
  }
  return shardMeta;
}

bool Graph::BuildSubEdgeSampler() {
  for (auto& it_vertex : vertex_map_) {
    Vertex* vertex = it_vertex.second;
    if (!vertex->BuildSubEdgeSampler()) {
      LOG(INFO) << " Vertex(" << vertex->GetId() << ") had not edge";
    }
  }
  return true;
}

bool Graph::BuildGlobalVertexSampler() {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  size_t vtype_num = static_cast<size_t>(schema->GetVTypeNum());
  std::vector<std::vector<Vertex*>> vertices(vtype_num);
  for (size_t i = 0; i < vertices.size(); ++i) {
    vertices[i].reserve(vertex_type_counts_[i]);
  }
  std::vector<float> vertex_type_weight_sum;
  std::vector<uint8_t> vertex_types;
  vertex_type_weight_sum.resize(vtype_num, 0.f);
  vertex_types.resize(vtype_num, -1);
  for (auto& vertex : vertex_map_) {
    int32_t vertex_type = vertex.second->GetType();
    vertices[vertex_type].push_back(vertex.second);
    vertex_type_weight_sum[vertex_type] += vertex.second->GetWeight();
  }
  vertex_samplers_.resize(vtype_num);
  for (size_t i = 0; i < vtype_num; ++i) {
    vertex_types[i] = static_cast<uint8_t>(i);
    if (!vertex_samplers_[i].Init(vertices[i])) {
      return false;
    }
  }

  if (!vertex_type_samplers_.Init(vertex_types, vertex_type_weight_sum)) {
    return false;
  }
  return true;
}

bool Graph::BuildGlobalEdgeSampler() {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  size_t etype_num = static_cast<size_t>(schema->GetETypeNum());
  std::vector<std::vector<Edge*>> edges(etype_num);
  for (size_t i = 0; i < edges.size(); ++i) {
    edges[i].reserve(edge_type_counts_[i]);
  }

  std::vector<float> edge_weight_sums;
  std::vector<uint8_t> edge_types;
  edge_weight_sums.resize(etype_num, 0);
  edge_types.resize(etype_num, 0);
  for (auto& it : edge_map_) {
    uint8_t e_type = it.second->GetType();
    edges[e_type].push_back(it.second);
    edge_weight_sums[e_type] += it.second->GetWeight();
  }

  edge_samplers_.resize(etype_num);
  for (size_t i = 0; i < etype_num; ++i) {
    edge_types[i] = static_cast<uint8_t>(i);
    if (!edge_samplers_[i].Init(edges[i])) {
      return false;
    }
  }

  if (!edge_type_samplers_.Init(edge_types, edge_weight_sums)) {
    return false;
  }
  return true;
}

#define RELEASE_VEC(VEC, nore)                                  \
  {                                                             \
    if (!VEC.empty()) {                                         \
      LOG(WARNING) << std::to_string(VEC.size()) << " " << nore \
                   << " have been ignored!";                    \
    }                                                           \
    while (!VEC.empty()) {                                      \
      auto it = VEC.end();                                      \
      VEC.erase(--it);                                          \
      if (*it) delete *it;                                      \
    }                                                           \
    VEC.clear();                                                \
  }

bool Graph::AddVertexs(std::vector<Vertex*>& vec) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  std::vector<Vertex*> invalid_vertices;
  for (auto& it : vec) {
    uint8_t type = it->GetType();
    if (unlikely(type >= schema->GetVTypeNum() ||
                 vertex_map_.find(it->GetId()) != vertex_map_.end())) {
      invalid_vertices.emplace_back(it);
    } else {
      vertex_map_[it->GetId()] = it;
      vertex_type_counts_[type]++;
      vertex_sum_weight_[type] += it->GetWeight();
    }
  }
  RELEASE_VEC(invalid_vertices, "vertices");
  return true;
}

bool Graph::AddEdges(std::vector<Edge*>& vec) {
  std::vector<Edge*> invalid_edges;
  for (auto& it : vec) {
    const galileo::common::VertexID src_id = it->GetSrcVertex();
    auto it_vertex = vertex_map_.find(src_id);
    if (unlikely(it_vertex == vertex_map_.end())) {
      invalid_edges.emplace_back(it);
    } else {
      if (!it_vertex->second->AddEdge(it)) {
        invalid_edges.emplace_back(it);
      } else {
        edge_type_counts_[it->GetType()]++;
        edge_sum_weight_[it->GetType()] += it->GetWeight();
        edge_map_.insert({it->GetId(), it});
      }
    }
  }
  RELEASE_VEC(invalid_edges, "edges");
  return true;
}

bool Graph::SampleVertex(const galileo::common::EntityRequest& entity_request,
                         galileo::common::Packer* packer) {
  size_t empty_cnt = 0;
  if (vertex_map_.size() <= 0) {
    LOG(ERROR) << " Please load vertex data first";
    return false;
  } else if (0 == entity_request.types_.cnt &&
             1 == entity_request.counts_.cnt) {
    // global sample
    size_t n_type_count(1);
    packer->Pack(n_type_count);
    size_t vertex_cnt = static_cast<size_t>(entity_request.counts_.data[0]);
    size_t offset = packer->Pack(vertex_cnt);
    for (size_t i = 0; i < vertex_cnt; ++i) {
      uint8_t n_type = vertex_type_samplers_.Sample();
      if (n_type >= static_cast<uint8_t>(vertex_samplers_.size())) {
        LOG(ERROR) << " Type sample error";
        packer->PackWithOffset(offset, empty_cnt);
        break;
      }
      std::pair<galileo::common::VertexID, float> sample_vertex(
          (galileo::common::VertexID)(-1), 0.f);
      bool ret = vertex_samplers_[n_type].Sample(sample_vertex);
      if (!ret) {
        LOG(ERROR) << " Can not find the vertex_type of " << n_type;
        packer->PackWithOffset(offset, empty_cnt);
        break;
      }
      // add the default vertex
      packer->Pack(sample_vertex.first);
    }
  } else if (entity_request.types_.cnt == entity_request.counts_.cnt &&
             entity_request.types_.cnt != 0) {
    packer->Pack(entity_request.types_.cnt);
    for (size_t i = 0; i < entity_request.types_.cnt; ++i) {
      uint8_t type = entity_request.types_.data[i];
      uint32_t n_count = entity_request.counts_.data[i];
      _SampleVertex(type, n_count, packer);
    }
  } else {
    LOG(ERROR) << "Params error: entity_request.types_.cnt = "
               << entity_request.types_.cnt << " ,entity_request.counts_.cnt = "
               << entity_request.counts_.cnt;
    return false;
  }
  return true;
}

bool Graph::SampleEdge(const galileo::common::EntityRequest& entity_request,
                       galileo::common::Packer* packer) {
  size_t empty_cnt = 0;
  if (edge_map_.size() <= 0) {
    LOG(ERROR) << " Please load edge data first";
    return false;
  } else if (0 == entity_request.types_.cnt &&
             1 == entity_request.counts_.cnt) {
    size_t e_type_count(1);
    packer->Pack(e_type_count);
    size_t edge_cnt = static_cast<size_t>(entity_request.counts_.data[0]);
    size_t offset = packer->Pack(edge_cnt);
    for (size_t i = 0; i < edge_cnt; ++i) {
      uint8_t e_type = edge_type_samplers_.Sample();
      if (e_type >= static_cast<uint8_t>(edge_samplers_.size())) {
        packer->PackWithOffset(offset, empty_cnt);
        LOG(ERROR) << " Type sample error";
        break;
      }
      std::pair<galileo::common::EdgeIDPtr, float> sample_edge;
      bool ret = edge_samplers_[e_type].Sample(sample_edge);
      if (!ret) {
        LOG(ERROR) << " Can not find the edge_type of " << e_type;
        packer->PackWithOffset(offset, empty_cnt);
        break;
      }
      packer->Pack(sample_edge.first.ptr->edge_type,
                   sample_edge.first.ptr->src_id,
                   sample_edge.first.ptr->dst_id);
    }
  } else if (entity_request.types_.cnt == entity_request.counts_.cnt &&
             entity_request.types_.cnt != 0) {
    packer->Pack(entity_request.types_.cnt);
    for (size_t i = 0; i < entity_request.types_.cnt; ++i) {
      uint8_t type = entity_request.types_.data[i];
      uint32_t e_count = entity_request.counts_.data[i];
      _SampleEdge(type, e_count, packer);
    }
  } else {
    LOG(ERROR) << " Params error: entity_request.types_.cnt = "
               << entity_request.types_.cnt << ", entity_request.counts_.cnt = "
               << entity_request.counts_.cnt;
    return false;
  }
  return true;
}

bool Graph::QueryNeighbors(
    galileo::common::OperatorType type,
    const galileo::common::NeighborRequest& neighbor_request,
    galileo::common::Packer* packer) {
  size_t empty_cnt = 0;
  packer->Pack(neighbor_request.ids_.cnt);
  for (size_t i = 0; i < neighbor_request.ids_.cnt; ++i) {
    galileo::common::VertexID n_id =
        (reinterpret_cast<const galileo::common::VertexID*>(
            neighbor_request.ids_.data))[i];
    Vertex* vertex = GetVertexByID(n_id);
    if (vertex == nullptr) {
      packer->Pack(empty_cnt);
      LOG(ERROR) << " Can not find vertex(" << n_id << ")";
      continue;
    }
    if (type == galileo::common::OperatorType::SAMPLE_NEIGHBOR) {
      vertex->SampleNeighbor(neighbor_request, packer);
    } else if (type == galileo::common::OperatorType::GET_TOPK_NEIGHBOR) {
      vertex->GetTopKNeighbor(neighbor_request, packer);
    } else if (type == galileo::common::OperatorType::GET_NEIGHBOR) {
      vertex->GetFullNeighbor(neighbor_request, packer);
    } else {
      LOG(ERROR) << " Input the sample type error :" << type;
      return false;
    }
  }
  return true;
}

bool Graph::GetEdgeFeature(
    const galileo::common::EdgeFeatureRequest& edge_feature_request,
    galileo::common::Packer* packer) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  size_t empty_cnt = 0;
  if (edge_feature_request.features_.size() !=
      edge_feature_request.max_dims_.cnt) {
    LOG(ERROR) << "_GetEdgeFeature input param error:"
               << "edge_feature_request.features_.size()="
               << edge_feature_request.features_.size() << ","
               << "edge_feature_request.max_dims_.cnt="
               << edge_feature_request.max_dims_.cnt;
    return false;
  }

  std::vector<std::string> feature_names(edge_feature_request.features_.size());
  std::vector<galileo::proto::DataType> feature_types;
  feature_types.resize(edge_feature_request.features_.size(),
                       galileo::proto::DT_INVALID_TYPE);

  for (size_t i = 0; i < edge_feature_request.features_.size(); ++i) {
    std::string& feature_name = feature_names[i];
    feature_name.resize(edge_feature_request.features_[i].cnt);
    std::copy(edge_feature_request.features_[i].data,
              edge_feature_request.features_[i].data +
                  edge_feature_request.features_[i].cnt,
              feature_name.begin());
  }

  packer->Pack(edge_feature_request.ids_.cnt);
  for (size_t i = 0; i < edge_feature_request.ids_.cnt; ++i) {
    galileo::common::EdgeID e_id =
        (reinterpret_cast<const galileo::common::EdgeID*>(
            edge_feature_request.ids_.data))[i];

    Edge* edge = GetEdgeByID(e_id);
    if (edge == nullptr) {
      packer->Pack(empty_cnt);
      LOG(ERROR) << " Can not find edge , src_id = " << e_id.src_id
                 << " ,dst_id = " << e_id.dst_id
                 << " ,edge_type=" << std::to_string(e_id.edge_type);
      continue;
    }
    packer->Pack(feature_names.size());
    uint8_t etype = edge->GetType();
    for (size_t idx = 0; idx < feature_names.size(); ++idx) {
      const std::string& feature_name = feature_names[idx];
      const char* attr_content = edge->GetFeature(feature_name);
      if (attr_content == nullptr) {
        packer->Pack(empty_cnt);
        feature_types[idx] = galileo::proto::DT_INVALID_TYPE;
        LOG(ERROR) << " Can not find edge feature , src_id = " << e_id.src_id
                   << " ,dst_id = " << e_id.dst_id
                   << " ,edge_type=" << e_id.edge_type
                   << " ,feature name=" << feature_name;
        continue;
      }
      int tmp_field_idx = schema->GetEFieldIdx(etype, feature_name);
      if (tmp_field_idx < 0) {
        LOG(ERROR) << " Can not find edge feature idx.etype:"
                   << std::to_string(etype) << ",feature_name:" << feature_name;
        return false;
      }
      size_t field_idx = static_cast<size_t>(tmp_field_idx);
      size_t field_len = schema->GetEFieldLen(etype, feature_name, true);
      const std::string& e_type_name = schema->GetEFieldDtype(etype, field_idx);
      galileo::proto::DataType eu_e_type =
          transformDataTypeByStrName(e_type_name);
      if (eu_e_type == galileo::proto::DT_INVALID_TYPE) {
        LOG(ERROR) << " Type can not support, src_id = " << e_id.src_id
                   << " ,dst_id = " << e_id.dst_id
                   << " ,edge_type=" << e_id.edge_type
                   << " ,type name=" << e_type_name;
        return false;
      }
      feature_types[idx] = eu_e_type;

      if (schema->IsEVarField(etype, field_idx)) {
        galileo::common::ArraySpec<char> array_value;
        uint16_t attr_len = *reinterpret_cast<const uint16_t*>(attr_content);
        if (eu_e_type == galileo::proto::DT_STRING) {
          array_value.cnt = static_cast<size_t>(attr_len);
          array_value.data = attr_content + sizeof(uint16_t);
        } else {
          attr_len = *reinterpret_cast<const uint16_t*>(attr_content +
                                                        sizeof(uint16_t));
          uint32_t dim = edge_feature_request.max_dims_.data[idx];
          if (dim > 0 && dim < static_cast<uint32_t>(attr_len)) {
            attr_len = static_cast<uint16_t>(dim);
          }
          array_value.cnt = static_cast<size_t>(attr_len) * field_len;
          array_value.data = attr_content + sizeof(uint16_t) + sizeof(uint16_t);
        }
        packer->Pack(array_value);
      } else {
        galileo::common::ArraySpec<char> array_value;
        array_value.cnt = field_len;
        array_value.data = attr_content;
        packer->Pack(array_value);
      }
    }
  }
  galileo::common::ArraySpec<galileo::proto::DataType> pk_features_type;
  pk_features_type.cnt = feature_types.size();
  pk_features_type.data = feature_types.data();
  packer->Pack(pk_features_type);
  return true;
}

bool Graph::GetVertexFeature(
    const galileo::common::VertexFeatureRequest& vertex_feature_request,
    galileo::common::Packer* packer) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  size_t empty_cnt = 0;
  if (vertex_feature_request.features_.size() !=
      vertex_feature_request.max_dims_.cnt) {
    std::ostringstream ostr;
    ostr << "_GetEdgeFeature input param error:"
         << "vertex_feature_request.features_.size()="
         << vertex_feature_request.features_.size() << ","
         << "vertex_feature_request.max_dims_.cnt="
         << vertex_feature_request.max_dims_.cnt;
    LOG(ERROR) << ostr.str();
    return false;
  }

  std::vector<std::string> feature_names(
      vertex_feature_request.features_.size());
  std::vector<galileo::proto::DataType> feature_types;
  feature_types.resize(vertex_feature_request.features_.size(),
                       galileo::proto::DT_INVALID_TYPE);

  for (size_t i = 0; i < vertex_feature_request.features_.size(); ++i) {
    std::string& feature_name = feature_names[i];
    feature_name.resize(vertex_feature_request.features_[i].cnt);
    std::copy(vertex_feature_request.features_[i].data,
              vertex_feature_request.features_[i].data +
                  vertex_feature_request.features_[i].cnt,
              feature_name.begin());
  }

  packer->Pack(vertex_feature_request.ids_.cnt);
  for (size_t i = 0; i < vertex_feature_request.ids_.cnt; ++i) {
    galileo::common::VertexID n_id =
        (reinterpret_cast<const galileo::common::VertexID*>(
            vertex_feature_request.ids_.data))[i];
    Vertex* vertex = GetVertexByID(n_id);
    if (vertex == nullptr) {
      packer->Pack(empty_cnt);
      LOG(ERROR) << " Can not find vertex , id = " << n_id;
      continue;
    }
    packer->Pack(feature_names.size());
    uint8_t vtype = vertex->GetType();
    for (size_t idx = 0; idx < feature_names.size(); ++idx) {
      const std::string& feature_name = feature_names[idx];
      const char* attr_content = vertex->GetFeature(feature_name);
      if (attr_content == nullptr) {
        packer->Pack(empty_cnt);
        feature_types[idx] = galileo::proto::DT_INVALID_TYPE;
        LOG(ERROR) << " Can not find vertex feature, vertex id = " << n_id
                   << ", feature name=" << feature_name;
        continue;
      }
      int tmp_idx = schema->GetVFieldIdx(vtype, feature_name);
      if (tmp_idx < 0) {
        LOG(ERROR) << " Get vertex feature idx fail. vertex type:"
                   << std::to_string(vtype) << ", filed name:" << feature_name;
        return false;
      }
      size_t field_idx = static_cast<size_t>(tmp_idx);
      size_t field_len = schema->GetVFieldLen(vtype, feature_name, true);
      const std::string& v_type_name = schema->GetVFieldDtype(vtype, field_idx);
      galileo::proto::DataType eu_v_type =
          transformDataTypeByStrName(v_type_name);
      if (eu_v_type == galileo::proto::DT_INVALID_TYPE) {
        LOG(ERROR) << " Type can not support, vertex id = " << n_id
                   << ", type name=" << v_type_name;
        return false;
      }
      feature_types[idx] = eu_v_type;

      if (schema->IsVVarField(vtype, field_idx)) {
        galileo::common::ArraySpec<char> array_value;
        uint16_t attr_len = *reinterpret_cast<const uint16_t*>(attr_content);
        if (eu_v_type == galileo::proto::DT_STRING) {
          array_value.cnt = attr_len;
          array_value.data = attr_content + sizeof(uint16_t);
        } else {
          attr_len = *reinterpret_cast<const uint16_t*>(attr_content +
                                                        sizeof(uint16_t));
          uint32_t dim = vertex_feature_request.max_dims_.data[idx];
          if (dim > 0 && dim < static_cast<uint32_t>(attr_len)) {
            attr_len = static_cast<uint16_t>(dim);
          }
          array_value.cnt = static_cast<size_t>(attr_len * field_len);
          array_value.data = attr_content + sizeof(uint16_t) + sizeof(uint16_t);
        }
        packer->Pack(array_value);
      } else {
        galileo::common::ArraySpec<char> array_value;
        array_value.cnt = field_len;
        array_value.data = attr_content;
        packer->Pack(array_value);
      }
    }
  }
  galileo::common::ArraySpec<galileo::proto::DataType> pk_features_type;
  pk_features_type.cnt = feature_types.size();
  pk_features_type.data = feature_types.data();
  packer->Pack(pk_features_type);
  return true;
}

void Graph::_SampleVertex(uint8_t vertex_type, uint32_t count,
                          galileo::common::Packer* packer) {
  size_t empty_cnt = 0;
  if (vertex_type >= vertex_samplers_.size()) {
    packer->Pack(empty_cnt);
    LOG(ERROR) << " Can not find the type of " << std::to_string(vertex_type);
    return;
  }
  size_t invalid_cnt = static_cast<size_t>(count);
  size_t offset = packer->Pack(invalid_cnt);
  for (size_t i = 0; i < invalid_cnt; ++i) {
    std::pair<galileo::common::VertexID, float> sample_vertex(
        (galileo::common::VertexID)(-1), 0.f);
    bool ret = vertex_samplers_[vertex_type].Sample(sample_vertex);
    if (!ret) {
      LOG(ERROR) << " Can not find the vertex_type of " << vertex_type;
      packer->PackWithOffset(offset, empty_cnt);
      return;
    }
    // add the default vertex
    packer->Pack(sample_vertex.first);
  }
}

void Graph::_SampleEdge(uint8_t edge_type, uint32_t count,
                        galileo::common::Packer* packer) {
  size_t empty_cnt = 0;
  if (edge_type >= edge_samplers_.size()) {
    packer->Pack(empty_cnt);
    LOG(ERROR) << " Can not find the type of " << std::to_string(edge_type);
    return;
  }
  size_t invalid_cnt = static_cast<size_t>(count);
  size_t offset = packer->Pack(invalid_cnt);
  for (size_t i = 0; i < invalid_cnt; ++i) {
    std::pair<galileo::common::EdgeIDPtr, float> sample_edge;
    bool ret = edge_samplers_[edge_type].Sample(sample_edge);
    if (!ret) {
      LOG(ERROR) << " Can not find the edge_type of " << edge_type;
      packer->PackWithOffset(offset, empty_cnt);
      return;
    }
    packer->Pack(sample_edge.first.ptr->edge_type);
    packer->Pack(sample_edge.first.ptr->src_id);
    packer->Pack(sample_edge.first.ptr->dst_id);
  }
}


galileo::proto::DataType transformDataTypeByStrName(const std::string& type) {
  if (type == "DT_INT8") {
    return galileo::proto::DT_INT8;
  } else if (type == "DT_UINT8") {
    return galileo::proto::DT_UINT8;
  } else if (type == "DT_INT16") {
    return galileo::proto::DT_INT16;
  } else if (type == "DT_UINT16") {
    return galileo::proto::DT_UINT16;
  } else if (type == "DT_INT32") {
    return galileo::proto::DT_INT32;
  } else if (type == "DT_UINT32") {
    return galileo::proto::DT_UINT32;
  } else if (type == "DT_INT64") {
    return galileo::proto::DT_INT64;
  } else if (type == "DT_UINT64") {
    return galileo::proto::DT_UINT64;
  } else if (type == "DT_BOOL") {
    return galileo::proto::DT_BOOL;
  } else if (type == "DT_FLOAT") {
    return galileo::proto::DT_FLOAT;
  } else if (type == "DT_DOUBLE") {
    return galileo::proto::DT_DOUBLE;
  } else if (type == "DT_STRING") {
    return galileo::proto::DT_STRING;
  } else if (type == "DT_ARRAY_INT8") {
    return galileo::proto::DT_ARRAY_INT8;
  } else if (type == "DT_ARRAY_UINT8") {
    return galileo::proto::DT_ARRAY_UINT8;
  } else if (type == "DT_ARRAY_INT32") {
    return galileo::proto::DT_ARRAY_INT32;
  } else if (type == "DT_ARRAY_UINT32") {
    return galileo::proto::DT_ARRAY_UINT32;
  } else if (type == "DT_ARRAY_INT64") {
    return galileo::proto::DT_ARRAY_INT64;
  } else if (type == "DT_ARRAY_UINT64") {
    return galileo::proto::DT_ARRAY_UINT64;
  } else if (type == "DT_ARRAY_BOOL") {
    return galileo::proto::DT_ARRAY_BOOL;
  } else if (type == "DT_ARRAY_FLOAT") {
    return galileo::proto::DT_ARRAY_FLOAT;
  } else if (type == "DT_ARRAY_DOUBLE") {
    return galileo::proto::DT_ARRAY_DOUBLE;
  }

  return galileo::proto::DT_INVALID_TYPE;
}
}  // namespace service
}  // namespace galileo
