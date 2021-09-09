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

#include <functional>
#include <memory>
#include <sstream>
#include <vector>

#include "glog/logging.h"

#include "client/dgraph_cutter.h"
#include "client/dgraph_shard.h"
#include "client/dgraph_type.h"

#include "common/message.h"
#include "common/packer.h"
#include "common/types.h"

#include "discovery/discoverer.h"
#include "proto/types.pb.h"

namespace galileo {
namespace client {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;
using EdgeArraySpec = galileo::common::EdgeArraySpec;

using CollectVertexRes = std::vector<std::vector<ArraySpec<VertexID>>>;
using CollectEdgeRes =
    std::vector<std::vector<ArraySpec<galileo::common::EdgeID>>>;
using CollectVertexTypeRes = std::vector<uint8_t>;

using CollectNeighborResWithWeight =
    std::vector<ArraySpec<galileo::common::IDWeight>>;
using CollectNeighborResWithoutWeight = std::vector<ArraySpec<VertexID>>;

struct CollectFeatureRes {
  std::vector<std::vector<ArraySpec<char>>> features_;
  ArraySpec<galileo::proto::DataType> features_type_;
};

class DGraphStub {
 public:
  DGraphStub();
  ~DGraphStub();

  bool Initialize(const DGraphConfig &config);

  template <typename ReplyType, typename ResType>
  void CollectEntity(
      galileo::common::OperatorType op, const ArraySpec<uint8_t> &types,
      uint32_t count,
      std::function<void(bool status, const ResType &res)> callback) const;

  template <typename ReplyType, typename ResType>
  void CollectNeighbor(
      galileo::common::OperatorType op, const ArraySpec<VertexID> &ids,
      const ArraySpec<uint8_t> &edges_type, uint32_t count, bool need_weight,
      std::function<void(bool status, ResType &res)> callback) const;

  void CollectFeature(
      galileo::common::OperatorType op, const char *ids,
      const std::vector<ArraySpec<char>> &features_name,
      const ArraySpec<uint32_t> &max_dims,
      std::function<void(bool status, CollectFeatureRes &res)> callback) const;

  bool CollectGraphMeta(GraphMeta *meta_info) const;

 private:
  void _SampleShard(uint32_t count, std::vector<uint32_t> &shards_idx,
                    std::vector<float> &shard_weight,
                    std::vector<uint32_t> *shards_count) const;

  void _AllocShardCount(
      const ArraySpec<uint8_t> &types, uint32_t count, WeightType weight_type,
      std::vector<std::vector<uint32_t>> *shards_counts) const;

  void _AllocShardEntity(const ArraySpec<VertexID> &ids,
                         std::vector<std::vector<size_t>> *shard_ids_idx) const;

  size_t _PackEntityRequest(const ArraySpec<uint8_t> &types,
                            std::vector<uint32_t> &count,
                            std::string *entity_req) const;

  size_t _PackNeighborRequest(const ArraySpec<VertexID> &ids,
                              const std::vector<size_t> &shard_ids_idx,
                              const ArraySpec<uint8_t> &edges_type,
                              uint32_t count, bool need_weight,
                              std::string *neighbor_req) const;

  size_t _PackFeatureRequest(galileo::common::OperatorType op, const char *ids,
                             const std::vector<size_t> &shard_ids_idx,
                             const std::vector<ArraySpec<char>> &features_name,
                             const ArraySpec<uint32_t> &max_dims,
                             std::string *feature_req) const;

  size_t _PackVertexShardInfo(const ArraySpec<VertexID> *vertices,
                              const std::vector<size_t> &shard_ids_idx,
                              galileo::common::Packer *packer) const;

  size_t _PackEdgeShardInfo(const EdgeArraySpec *edges,
                            const std::vector<size_t> &shard_ids_idx,
                            galileo::common::Packer *packer) const;

  size_t _PackVertexTypeRequest(const ArraySpec<VertexID> &ids,
                                const std::vector<size_t> &shard_ids_idx,
                                std::string *vertex_type_req) const;

  void _ConstructRpcReq(galileo::common::OperatorType op, std::string *request,
                        galileo::proto::QueryRequest *rpc_request) const;

  bool _IsValidFeatureType(
      ArraySpec<galileo::proto::DataType> &features_type) const;

 private:
  std::shared_ptr<galileo::discovery::Discoverer> discoverer_;
  DGraphShard *shards_;
  uint32_t shard_num_;
  DGraphCutter cutter_;
};

template <typename ReplyType, typename ResType>
void DGraphStub::CollectEntity(
    galileo::common::OperatorType op, const ArraySpec<uint8_t> &types,
    uint32_t count,
    std::function<void(bool status, const ResType &res)> callback) const {
  ResType *res = new ResType();
  if (op != galileo::common::SAMPLE_VERTEX &&
      op != galileo::common::SAMPLE_EDGE) {
    LOG(ERROR) << " The op param of CollectEntity is invalid.cur op value:"
               << op;
    callback(false, *res);
    delete res;
    return;
  }
  std::vector<std::vector<uint32_t>> shards_counts;
  if (galileo::common::SAMPLE_VERTEX == op) {
    this->_AllocShardCount(types, count, VERTEX_WEIGHT, &shards_counts);
  } else {
    this->_AllocShardCount(types, count, EDGE_WEIGHT, &shards_counts);
  }
  size_t type_num = 0 == types.cnt ? 1 : types.cnt;
  res->resize(shard_num_);
  std::atomic<uint32_t> *callback_num = new std::atomic<uint32_t>(0);
  std::atomic<bool> *status = new std::atomic<bool>(true);
  galileo::proto::QueryResponse *rpc_response =
      new galileo::proto::QueryResponse[shard_num_];
  auto shard_callback = [this, type_num, count, shards_counts, callback,
                         rpc_response, res, callback_num,
                         status](bool is_ok, uint32_t shard_id,
                                 std::string *response) {
    if (!is_ok) {
      *status = false;
    }
    if (*status) {
      ReplyType reply;
      galileo::common::Packer reply_packer(response);
      if (!reply_packer.UnPack(&reply.ids_)) {
        LOG(ERROR) << " Unpack entity reply fail.shard id" << shard_id;
        *status = false;
      }else{
        if (reply.ids_.size() != type_num) {
          LOG(ERROR) << " Reply ids size is invalid."
                   << " shard  id:" << shard_id
                   << " ,expected size:" << type_num
                   << " ,real size:" << reply.ids_.size();
          *status = false;
        }else{
          res->at(shard_id) = std::move(reply.ids_);
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
    this->_PackEntityRequest(types, shards_counts[shard_idx], &req_str);
    this->_ConstructRpcReq(op, &req_str, &rpc_request);
    shards_[shard_idx].Collect(rpc_request, &rpc_response[shard_idx],
                               shard_callback);
  }
}

template <typename ReplyType, typename ResType>
void DGraphStub::CollectNeighbor(
    galileo::common::OperatorType op, const ArraySpec<VertexID> &ids,
    const ArraySpec<uint8_t> &edges_type, uint32_t count, bool need_weight,
    std::function<void(bool status, ResType &res)> callback) const {
  ResType *res = new ResType();
  if (op != galileo::common::SAMPLE_NEIGHBOR &&
      op != galileo::common::GET_NEIGHBOR &&
      op != galileo::common::GET_TOPK_NEIGHBOR) {
    LOG(ERROR) << " The op param of CollectNeighbor is invalid .cur op value:"
               << op;
    callback(false, *res);
    delete res;
    return;
  }
  std::vector<std::vector<size_t>> shard_ids_idx;
  this->_AllocShardEntity(ids, &shard_ids_idx);
  std::atomic<uint32_t> *callback_num = new std::atomic<uint32_t>(0);
  std::atomic<bool> *status = new std::atomic<bool>(true);
  galileo::proto::QueryResponse *rpc_response =
      new galileo::proto::QueryResponse[shard_num_];
  res->resize(ids.cnt);
  auto shard_callback = [this, op, count, callback, rpc_response, res,
                         shard_ids_idx, callback_num,
                         status](bool is_ok, uint32_t shard_id,
                                 std::string *response) {
    if (!is_ok) {
      *status = false;
    }
    if (*status) {
      ReplyType reply;
      galileo::common::Packer reply_packer(response);
      if (!reply_packer.UnPack(&reply.neighbors_)) {
        LOG(ERROR) << " Unpack neighbor reply fail.shard_id:" << shard_id;
        *status = false;
      }else{
        auto &idx = shard_ids_idx[shard_id];
        if (idx.size() != reply.neighbors_.size()) {
          LOG(ERROR) << " Neighbor reply id size is invalid."
                   << " shard id:" << shard_id
                   << " ,expected size:" << idx.size()
                   << " ,real size:" << reply.neighbors_.size();
          *status = false;
        }else{
          for (size_t i = 0; i < reply.neighbors_.size(); ++i) {
            res->at(idx[i]) = reply.neighbors_[i];
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
    this->_PackNeighborRequest(ids, shard_ids_idx[shard_idx], edges_type, count,
                               need_weight, &req_str);
    this->_ConstructRpcReq(op, &req_str, &rpc_request);
    shards_[shard_idx].Collect(rpc_request, &rpc_response[shard_idx],
                               shard_callback);
  }
}

}  // namespace client
}  // namespace galileo
