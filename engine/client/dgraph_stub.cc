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

#include <assert.h>
#include <atomic>
#include <string>
#include <vector>

#include "client/dgraph_stub.h"
#include "common/sampler.h"
#include "common/types.h"

namespace galileo {
namespace client {

DGraphStub::DGraphStub()
    : discoverer_(nullptr), shards_(nullptr), shard_num_(0) {}

DGraphStub::~DGraphStub() {
  if (shards_ != nullptr) {
    delete[] shards_;
    shards_ = nullptr;
  }
}

bool DGraphStub::Initialize(const DGraphConfig &config) {
  LOG(INFO) << " Initialize dgraph stub start";
  LOG(INFO) << " Zk info:zk addr:" << config.zk_addr
            << ",zk_path:" << config.zk_path;
  discoverer_ = std::make_shared<galileo::discovery::Discoverer>(
      config.zk_addr, config.zk_path);

  shard_num_ = discoverer_->GetShardsNum();

  uint32_t partition_num = discoverer_->GetPartitionsNum();

  LOG(INFO) << " Meta info:[shard_num:" << shard_num_
            << " ,partition_num:" << partition_num << "].";
  cutter_.Reset(partition_num, shard_num_);

  shards_ = new DGraphShard[shard_num_];
  for (uint32_t shard_idx = 0; shard_idx < shard_num_; ++shard_idx) {
    shards_[shard_idx].Initilize(shard_idx, discoverer_, config);
  }
  LOG(INFO) << " Initialize dgraph stub finish";
  return true;
}

void DGraphStub::CollectFeature(
    galileo::common::OperatorType op, const char *ids,
    const std::vector<ArraySpec<char>> &features_name,
    const ArraySpec<uint32_t> &max_dims,
    std::function<void(bool status, CollectFeatureRes &res)> callback) const {
  CollectFeatureRes *res = new CollectFeatureRes();
  if (op != galileo::common::GET_VERTEX_FEATURE &&
      op != galileo::common::GET_EDGE_FEATURE) {
    LOG(ERROR) << " The op param of CollectFeature is invalid .cur op value:"
               << op;
    callback(false, *res);
    delete res;
    return;
  }
  size_t id_cnt = 0;
  std::vector<std::vector<size_t>> shard_ids_idx;
  if (galileo::common::GET_VERTEX_FEATURE == op) {
    const ArraySpec<VertexID> *vertices = (const ArraySpec<VertexID> *)ids;
    id_cnt = vertices->cnt;
    this->_AllocShardEntity(*vertices, &shard_ids_idx);
  } else {
    const EdgeArraySpec *edges = (const EdgeArraySpec *)ids;
    id_cnt = edges->srcs.cnt;
    this->_AllocShardEntity(edges->srcs, &shard_ids_idx);
  }
  galileo::proto::QueryResponse *rpc_response =
      new galileo::proto::QueryResponse[shard_num_];
  std::atomic<uint32_t> *callback_num = new std::atomic<uint32_t>(0);
  std::atomic<bool> *status = new std::atomic<bool>(true);
  res->features_.resize(id_cnt);
  res->features_type_.cnt = 0;
  size_t feature_num = features_name.size();
  auto shard_callback = [this, feature_num, callback, rpc_response, res,
                         shard_ids_idx, callback_num,
                         status](bool is_ok, uint32_t shard_id,
                                 std::string *response) {
    if (!is_ok) {
      *status = false;
    }
    if (*status) {
      galileo::common::FeatureReply reply;
      galileo::common::Packer reply_packer(response);
      if (!reply_packer.UnPack(&reply.features_, &reply.features_type_)) {
        LOG(ERROR) << "Unpack feature reply fail. shard id:" << shard_id;
        *status = false;
      } else {
        if (reply.features_type_.cnt != feature_num) {
          LOG(ERROR) << " Features_type num of feature reply is invalid."
                     << " shard id:" << shard_id
                     << " ,expected num:" << feature_num
                     << " ,real num:" << reply.features_type_.cnt;
          *status = false;
        } else {
          if (feature_num != res->features_type_.cnt &&
              this->_IsValidFeatureType(reply.features_type_)) {
            res->features_type_ = reply.features_type_;
          }
          auto &idx = shard_ids_idx[shard_id];
          if (idx.size() != reply.features_.size()) {
            LOG(ERROR) << " Feature reply id size is invalid."
                       << " shard id:" << shard_id
                       << " ,expected size:" << idx.size()
                       << " ,real size:" << reply.features_.size();
            *status = false;
          } else {
            for (size_t i = 0; i < idx.size(); ++i) {
              for (size_t j = 0; j < reply.features_[i].size(); ++j) {
                res->features_[idx[i]].push_back(reply.features_[i][j]);
              }
            }
          }
        }
      }
    }
    auto callbacks = ++*callback_num;
    if (callbacks == shard_num_) {
      callback(*status, *res);
      delete res;
      delete[] rpc_response;
      delete callback_num;
      delete status;
    }
  };
  for (uint32_t shard_idx = 0; shard_idx < shard_num_; ++shard_idx) {
    std::string req_str;
    galileo::proto::QueryRequest rpc_request;
    this->_PackFeatureRequest(op, ids, shard_ids_idx[shard_idx], features_name,
                              max_dims, &req_str);
    this->_ConstructRpcReq(op, &req_str, &rpc_request);
    shards_[shard_idx].Collect(rpc_request, &rpc_response[shard_idx],
                               shard_callback);
  }
}

bool DGraphStub::CollectGraphMeta(GraphMeta *meta_info) const {
  meta_info->vertex_size = discoverer_->GetVertexSize();
  meta_info->edge_size = discoverer_->GetEdgeSize();
  return true;
}

bool DGraphStub::_IsValidFeatureType(
    ArraySpec<proto::DataType> &features_type) const {
  for (size_t i = 0; i < features_type.cnt; ++i) {
    if (galileo::proto::DT_INVALID_TYPE != features_type.data[i]) {
      return true;
    }
  }
  return false;
}

void DGraphStub::_ConstructRpcReq(
    galileo::common::OperatorType op, std::string *request,
    galileo::proto::QueryRequest *rpc_request) const {
  rpc_request->set_op_type(op);
  rpc_request->set_data(std::move(*request));
}

void DGraphStub::_SampleShard(uint32_t count, std::vector<uint32_t> &shards_idx,
                              std::vector<float> &shard_weight,
                              std::vector<uint32_t> *shards_count) const {
  shards_count->resize(shard_num_, 0);
  galileo::common::SimpleSampler<uint32_t> sampler;
  sampler.Init(shards_idx, shard_weight);
  for (uint32_t i = 0; i < count; i++) {
    uint32_t shard_idx = sampler.Sample();
    ++shards_count->at(shard_idx);
  }
}

void DGraphStub::_AllocShardCount(
    const ArraySpec<uint8_t> &types, uint32_t count, WeightType weight_type,
    std::vector<std::vector<uint32_t>> *shards_counts) const {
  shards_counts->resize(shard_num_);
  std::vector<uint32_t> shards_idx;
  shards_idx.reserve(shard_num_);
  for (uint32_t i = 0; i < shard_num_; ++i) {
    shards_idx.push_back(i);
  }
  size_t t_idx = 0;
  do {
    std::vector<float> shards_weight;
    shards_weight.reserve(shard_num_);
    int type_idx = 0 == types.cnt ? -1 : static_cast<int>(types.data[t_idx]);
    for (uint32_t i = 0; i < shard_num_; ++i) {
      shards_weight.push_back(shards_[i].GetWeight(weight_type, type_idx));
    }
    std::vector<uint32_t> shards_cnt;
    this->_SampleShard(count, shards_idx, shards_weight, &shards_cnt);
    for (uint32_t shard_idx = 0; shard_idx < shard_num_; ++shard_idx) {
      auto &cnt = shards_counts->at(shard_idx);
      cnt.push_back(shards_cnt[shard_idx]);
    }
    ++t_idx;
  } while (t_idx < types.cnt);
}

void DGraphStub::_AllocShardEntity(
    const ArraySpec<VertexID> &ids,
    std::vector<std::vector<size_t>> *shard_ids_idx) const {
  shard_ids_idx->resize(shard_num_);
  for (size_t i = 0; i < ids.cnt; ++i) {
    uint32_t shard_idx = cutter_.IDCut(ids.data[i]);
    auto &indexes = shard_ids_idx->at(shard_idx);
    indexes.push_back(i);
  }
}

size_t DGraphStub::_PackEntityRequest(const ArraySpec<uint8_t> &types,
                                      std::vector<uint32_t> &counts,
                                      std::string *entity_req) const {
  size_t req_len =
      types.Capacity() + sizeof(size_t) + sizeof(uint32_t) * counts.size();
  galileo::common::Packer entity_packer(entity_req, req_len);
  entity_packer.Pack(types, counts);
  return entity_packer.PackEnd();
}

size_t DGraphStub::_PackNeighborRequest(
    const ArraySpec<VertexID> &ids, const std::vector<size_t> &shard_ids_idx,
    const ArraySpec<uint8_t> &edges_type, uint32_t count, bool need_weight,
    std::string *neighbor_req) const {
  size_t shard_id_num = shard_ids_idx.size();
  size_t req_len = sizeof(shard_id_num) + sizeof(VertexID) * shard_id_num +
                   edges_type.Capacity() + sizeof(count) + sizeof(need_weight);
  galileo::common::Packer neighbor_packer(neighbor_req, req_len);
  neighbor_packer.Pack(shard_id_num);
  for (auto &id_idx : shard_ids_idx) {
    neighbor_packer.Pack(ids.data[id_idx]);
  }
  neighbor_packer.Pack(edges_type, count, need_weight);
  return neighbor_packer.PackEnd();
}

size_t DGraphStub::_PackVertexShardInfo(
    const ArraySpec<VertexID> *vertices,
    const std::vector<size_t> &shard_ids_idx,
    galileo::common::Packer *packer) const {
  size_t offset = 0;
  for (auto &id_idx : shard_ids_idx) {
    offset = packer->Pack(vertices->data[id_idx]);
  }
  return offset;
}

size_t DGraphStub::_PackEdgeShardInfo(const EdgeArraySpec *edges,
                                      const std::vector<size_t> &shard_ids_idx,
                                      galileo::common::Packer *packer) const {
  size_t offset = 0;
  for (auto &id_idx : shard_ids_idx) {
    offset = packer->Pack(edges->types.data[id_idx]);
    offset = packer->Pack(edges->srcs.data[id_idx]);
    offset = packer->Pack(edges->dsts.data[id_idx]);
  }
  return offset;
}

size_t DGraphStub::_PackFeatureRequest(
    galileo::common::OperatorType op, const char *ids,
    const std::vector<size_t> &shard_ids_idx,
    const std::vector<ArraySpec<char>> &features_name,
    const ArraySpec<uint32_t> &max_dims, std::string *feature_req) const {
  size_t shard_id_num = shard_ids_idx.size();
  galileo::common::Packer feature_packer(feature_req);
  feature_packer.Pack(shard_id_num);
  if (galileo::common::GET_VERTEX_FEATURE == op) {
    const ArraySpec<VertexID> *vertices = (const ArraySpec<VertexID> *)ids;
    this->_PackVertexShardInfo(vertices, shard_ids_idx, &feature_packer);
  } else {
    const EdgeArraySpec *edges = (const EdgeArraySpec *)ids;
    this->_PackEdgeShardInfo(edges, shard_ids_idx, &feature_packer);
  }
  feature_packer.Pack(features_name, max_dims);
  return feature_packer.PackEnd();
}

}  // namespace client
}  // namespace galileo
