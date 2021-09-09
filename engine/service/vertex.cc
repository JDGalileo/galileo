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

#include "service/vertex.h"

#include <assert.h>
#include <float.h>
#include <algorithm>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/macro.h"
#include "common/schema.h"
#include "common/singleton.h"
#include "glog/logging.h"
#include "service/edge.h"
#include "service/graph.h"
#include "utils/bytes_reader.h"

namespace galileo {
namespace service {

using Schema = galileo::schema::Schema;

Vertex::Vertex(uint8_t vtype) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  raw_data_ = galileo::common::Singleton<EntityPoolManager>::GetInstance()
                  ->GetVMemoryPool(vtype)
                  ->NextRecord();
  memset(raw_data_, 0, schema->GetVRecordSize(vtype));

  edge_counts_.resize(static_cast<size_t>(schema->GetETypeNum()), 0);
}

Vertex::~Vertex() {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t vtype = GetType();
  for (size_t i = 0; i < schema->GetVFieldCount(vtype); ++i) {
    if (schema->IsVVarField(GetType(), i)) {
      std::string attr = schema->GetVFieldName(vtype, i);
      if (!_IsDirectStore(attr)) {
        int offset = schema->GetVFieldOffset(vtype, attr);
        assert(offset >= 0);
        uint64_t var_address =
            *reinterpret_cast<const uint64_t*>(raw_data_ + offset);
        char* real_data = (char*)var_address;
        if (real_data) free(real_data);
      }
    }
  }
}

galileo::common::VertexID Vertex::GetId() const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  int offset = schema->GetVFieldOffset(GetType(), SCM_ENTITY);
  assert(offset >= 0);
  return *reinterpret_cast<const galileo::common::VertexID*>(raw_data_ +
                                                             offset);
}

float Vertex::GetWeight() const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  int offset = schema->GetVFieldOffset(GetType(), SCM_WEIGHT);
  if (offset < 0) {
    return 1.0;
  } else {
    return *reinterpret_cast<const float*>(raw_data_ + offset);
  }
}

const char* Vertex::GetFeature(const std::string& attr_name) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t vtype = GetType();
  int tmp_fid = schema->GetVFieldIdx(vtype, attr_name);
  if (tmp_fid < 0) {
    return nullptr;
  }
  size_t fid = static_cast<size_t>(tmp_fid);
  int begin_idx = schema->GetVFieldOffset(vtype, attr_name);
  if (begin_idx < 0) {
    return nullptr;
  }
  if (schema->IsVVarField(vtype, fid) && !_IsDirectStore(attr_name)) {
    uint64_t var_address =
        *reinterpret_cast<const uint64_t*>(raw_data_ + begin_idx);
    return reinterpret_cast<const char*>(var_address);
  } else {
    return reinterpret_cast<const char*>(raw_data_ + begin_idx);
  }
}

bool Vertex::DeSerialize(uint8_t type, const char* s, size_t size) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  galileo::utils::BytesReader bytes_reader(s, size);
  size_t write_off = schema->GetVFixedFieldsLen(type);
  if (write_off > size) {
    LOG(ERROR) << " Fix attribute info error."
               << " type:" << std::to_string(type)
               << " ,write_off:" << write_off << " ,size:" << size;
    return false;
  }
  // fix attr
  if (!bytes_reader.Read(raw_data_, write_off)) {
    LOG(ERROR) << " Fix attribute info error";
    return false;
  }

  // var attr
  uint16_t var_len;
  for (size_t i = 0; i < schema->GetVVarFieldCount(type); i++) {
    if (!bytes_reader.Peek(&var_len)) {
      LOG(ERROR) << " Var attribute length error";
      return false;
    }
    if (_IsDirectStore(i)) {
      if (!bytes_reader.Read(raw_data_ + write_off, var_len + 2)) {
        LOG(ERROR) << " Var attribute info error";
        return false;
      }
    } else {
      char* var_attr = (char*)malloc(var_len + 2);
      if (!bytes_reader.Read(var_attr, var_len + 2)) {
        LOG(ERROR) << " Var attribute info error";
        return false;
      }
      uint64_t var_address = (uint64_t)var_attr;
      memcpy(raw_data_ + write_off, &var_address, 8);
    }
    write_off += 8;
  }
  return true;
}

bool Vertex::AddEdge(Edge* edge) {
  uint8_t etype = edge->GetType();
  auto it = out_edges_.find(etype);
  if (it == out_edges_.end()) {
    std::vector<Edge*>* edges = new std::vector<Edge*>();
    edges->emplace_back(edge);
    out_edges_[etype] = edges;
  } else {
    it->second->emplace_back(edge);
  }
  ++edge_counts_[etype];
  return true;
}

void Vertex::SampleNeighbor(
    const galileo::common::NeighborRequest& neighbor_request,
    galileo::common::Packer* packer) {
  size_t zero_val = 0;
  if (!_HadEdge()) {
    LOG(WARNING) << " The neighbor num of vertex is zero.vertex id:" << GetId();
    packer->Pack(zero_val);
    return;
  }
  std::vector<uint8_t> valid_types;
  bool sample_all = false;
  if (0 == neighbor_request.edge_types_.cnt) {
    sample_all = true;
  } else {
    for (size_t i = 0; i < neighbor_request.edge_types_.cnt; ++i) {
      uint8_t e_type = neighbor_request.edge_types_.data[i];
      if (e_type >= edge_samplers_.size() || edge_samplers_[e_type].IsEmpty()) {
        LOG(ERROR) << " Cannt find the type (" << std::to_string(e_type)
                   << ") in vertex(" << GetId() << ")";
      } else {
        valid_types.push_back(e_type);
      }
    }
    if (valid_types.size() <= 0) {
      LOG(ERROR) << " Cannt find valid_type in vertex(" << GetId() << ")";
      packer->Pack(zero_val);
      return;
    }
  }

  galileo::common::SimpleSampler<uint8_t> sub_edge_type_samplers;
  if (!sample_all) {
    std::vector<uint8_t> edge_types(valid_types.size());
    std::vector<float> edge_weights(valid_types.size());
    for (size_t i = 0; i < valid_types.size(); ++i) {
      uint8_t e_type = valid_types[i];
      edge_types[i] = e_type;
      edge_weights[i] = edge_type_samplers_.GetWeight(i);
    }
    sub_edge_type_samplers.Init(edge_types, edge_weights);
  }
  size_t expected_cnt = static_cast<size_t>(neighbor_request.cnt);
  size_t offset = packer->Pack(expected_cnt);
  for (size_t i = 0; i < expected_cnt; ++i) {
    if (sample_all) {
      galileo::common::EdgeType e_type = edge_type_samplers_.Sample();
      if (edge_samplers_[e_type].IsEmpty()) {
        LOG(ERROR) << " Can not find sample edge of type "
                   << std::to_string(e_type);
        packer->PackWithOffset(offset, zero_val);
        break;
      }
      std::pair<galileo::common::EdgeIDPtr, float> sample_edge;
      edge_samplers_[e_type].Sample(sample_edge);
      if (neighbor_request.need_weight_) {
        packer->Pack(sample_edge.first.ptr->dst_id, sample_edge.second);
      } else {
        packer->Pack(sample_edge.first.ptr->dst_id);
      }
    } else {
      uint8_t e_type = sub_edge_type_samplers.Sample();
      std::pair<galileo::common::EdgeIDPtr, float> sample_edge;
      edge_samplers_[e_type].Sample(sample_edge);
      if (neighbor_request.need_weight_) {
        packer->Pack(sample_edge.first.ptr->dst_id, sample_edge.second);
      } else {
        packer->Pack(sample_edge.first.ptr->dst_id);
      }
    }
  }
}

void Vertex::GetTopKNeighbor(
    const galileo::common::NeighborRequest& neighbor_request,
    galileo::common::Packer* packer) {
  size_t zero_val = 0;
  if (!_HadEdge()) {
    LOG(WARNING) << " The neighbor num of vertex is zero.vertex id:" << GetId();
    packer->Pack(zero_val);
    return;
  }
  std::vector<uint8_t> valid_types;
  if (0 == neighbor_request.edge_types_.cnt) {
    for (uint8_t i = 0; i < edge_samplers_.size(); ++i) {
      if (!edge_samplers_[i].IsEmpty()) {
        valid_types.push_back(i);
      }
    }
  } else {
    for (size_t i = 0; i < neighbor_request.edge_types_.cnt; ++i) {
      uint8_t e_type = neighbor_request.edge_types_.data[i];
      if (e_type >= edge_samplers_.size() || edge_samplers_[e_type].IsEmpty()) {
        LOG(ERROR) << " Cannt find the type (" << std::to_string(e_type)
                   << ") in vertex(" << GetId() << ")";
      } else {
        valid_types.push_back(e_type);
      }
    }
  }
  if (valid_types.size() <= 0) {
    packer->Pack(zero_val);
    return;
  }

  std::priority_queue<galileo::common::IDWeight,
                      std::vector<galileo::common::IDWeight>,
                      VertexWeightComparision>
      min_heap;
  for (size_t i = 0; i < valid_types.size(); ++i) {
    uint8_t e_type = valid_types[i];
    for (size_t j = 0; j < out_edges_[e_type]->size(); ++j) {
      Edge* edge = (*out_edges_[e_type])[j];
      if (min_heap.size() < static_cast<size_t>(neighbor_request.cnt)) {
        min_heap.push({edge->GetId().ptr->dst_id, edge->GetWeight()});
      } else {
        if (min_heap.top().weight_ < edge->GetWeight()) {
          min_heap.pop();
          min_heap.push({edge->GetId().ptr->dst_id, edge->GetWeight()});
        }
      }
    }
  }
  packer->Pack(min_heap.size());
  while (!min_heap.empty()) {
    packer->Pack(min_heap.top().id_);
    if (neighbor_request.need_weight_) {
      packer->Pack(min_heap.top().weight_);
    }
    min_heap.pop();
  }
}

void Vertex::GetFullNeighbor(
    const galileo::common::NeighborRequest& neighbor_request,
    galileo::common::Packer* packer) {
  size_t zero_val = 0;
  if (!_HadEdge()) {
    LOG(WARNING) << " The neighbor num of vertex is zero.vertex id:" << GetId();
    packer->Pack(zero_val);
    return;
  }
  std::vector<uint8_t> edge_types;
  if (0 == neighbor_request.edge_types_.cnt) {
    std::unordered_map<uint8_t, std::vector<Edge*>*>::iterator it =
        out_edges_.begin();
    for (; it != out_edges_.end(); ++it) {
      edge_types.push_back(it->first);
    }
  } else {
    for (size_t i = 0; i < neighbor_request.edge_types_.cnt; ++i) {
      uint8_t e_type = neighbor_request.edge_types_.data[i];
      std::unordered_map<uint8_t, std::vector<Edge*>*>::iterator it =
          out_edges_.find(e_type);
      if (it != out_edges_.end()) {
        edge_types.push_back(it->first);
      }
    }
  }
  size_t edge_count = 0;
  size_t offset = packer->Pack(edge_count);
  for (size_t i = 0; i < edge_types.size(); ++i) {
    uint8_t e_type = edge_types[i];
    std::vector<Edge*>* edges = out_edges_[e_type];
    for (size_t j = 0; j < edges->size(); ++j) {
      ++edge_count;
      Edge* edge = (*edges)[j];
      packer->Pack(edge->GetId().ptr->dst_id);
      if (neighbor_request.need_weight_) {
        packer->Pack(edge->GetWeight());
      }
    }
  }
  packer->PackWithOffset(offset, edge_count);
}

bool Vertex::BuildSubEdgeSampler() {
  size_t etype_num = edge_counts_.size();
  std::vector<std::vector<Edge*>> edges(etype_num);
  for (size_t i = 0; i < etype_num; ++i) {
    size_t edge_type_num = static_cast<size_t>(edge_counts_[i]);
    edges[i].reserve(edge_type_num);
  }
  std::vector<float> edge_weight_sums;
  std::vector<uint8_t> edge_types;
  edge_weight_sums.resize(etype_num, 0);
  edge_types.resize(etype_num);
  for (auto& it_edges : out_edges_) {
    edges[it_edges.first].insert(edges[it_edges.first].end(),
                                 it_edges.second->begin(),
                                 it_edges.second->end());
    for (auto& it_edge : *(it_edges.second)) {
      edge_weight_sums[it_edges.first] += (it_edge)->GetWeight();
    }
  }
  edge_samplers_.resize(etype_num);
  for (size_t i = 0; i < etype_num; ++i) {
    edge_types[i] = static_cast<uint8_t>(i);
    // need not check the result, because some type have not the edge
    edge_samplers_[i].Init(edges[i]);
  }

  if (!edge_type_samplers_.Init(edge_types, edge_weight_sums)) {
    return false;
  }
  return true;
}

bool Vertex::_HadEdge() {
  bool had_edge = false;
  for (size_t i = 0; i < edge_samplers_.size(); ++i) {
    if (!edge_samplers_[i].IsEmpty()) {
      had_edge = true;
      break;
    }
  }
  return had_edge;
}

bool Vertex::_IsDirectStore(const std::string& field_name) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t vtype = GetType();
  int var_index = schema->GetVVarFieldIdx(vtype, field_name);
  if (var_index < 0) {
    LOG(ERROR) << " Get vertex var field idx fail."
               << " vertex type:" << vtype << " ,field name:" << field_name;
    return false;
  }
  return _IsDirectStore(static_cast<size_t>(var_index));
}

bool Vertex::_IsDirectStore(size_t var_index) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  char state =
      *(raw_data_ + schema->GetVStateOffset(GetType()) + (var_index / 8));
  return state & (1 << (7 - (var_index % 8)));
}

}  // namespace service
}  // namespace galileo
