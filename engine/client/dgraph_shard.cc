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

#include <sstream>

#include "client/dgraph_shard.h"
#include "common/message.h"
#include "common/packer.h"

#include "proto/rpc.pb.h"

#include "glog/logging.h"

namespace galileo {
namespace client {

DGraphShard::DGraphShard()
    : rpc_client_(new galileo::rpc::Client()),
      id_(0),
      vertex_weights_(0.0),
      edge_weights_(0.0) {}

bool DGraphShard::Initilize(
    uint32_t id, std::shared_ptr<galileo::discovery::Discoverer> discoverer,
    const DGraphConfig &config) {
  LOG(INFO) << " Initialize dgraph shard start.";
  id_ = id;
  discoverer->GetVertexWeightSum(id_, &vertex_type_weight_);
  discoverer->GetEdgeWeightSum(id_, &edge_type_weight_);
  for (auto weight : vertex_type_weight_) {
    vertex_weights_ += weight;
  }
  for (auto weight : edge_type_weight_) {
    edge_weights_ += weight;
  }
  LOG(INFO) << " Initialize dgraph shard finish.";
  return rpc_client_->Init(id_, discoverer, config);
}

void DGraphShard::Collect(
    const galileo::proto::QueryRequest &rpc_request,
    galileo::proto::QueryResponse *rpc_response,
    std::function<void(bool is_ok, uint32_t shard_id, std::string *response)>
        callback) const {
  auto rpc_callback = [this, rpc_response, callback](bool is_ok) {
    std::string *tmp_reply = nullptr;
    if (is_ok) {
      tmp_reply = rpc_response->mutable_data();
      if (nullptr == tmp_reply) {
        LOG(ERROR) << "Rpc response is nullptr.shard_id:" << id_;
        callback(false, id_, tmp_reply);
      } else if (0 == tmp_reply->size()) {
        LOG(ERROR) << "Reply size is zero.shard_id:" << id_;
        callback(false, id_, tmp_reply);
      } else {
        callback(true, id_, tmp_reply);
      }
    } else {
      LOG(ERROR) << "Rpc callback fail.";
      callback(false, id_, tmp_reply);
    }
  };
  rpc_client_->Query(rpc_request, rpc_response, rpc_callback);
}

float DGraphShard::GetWeight(WeightType weight_type, int entity_type) {
  if (entity_type < 0) {
    return this->_GetWeightSum(weight_type);
  }
  float weight = 0.0;
  switch (weight_type) {
    case VERTEX_WEIGHT:
      weight = vertex_type_weight_[entity_type];
      break;
    case EDGE_WEIGHT:
      weight = edge_type_weight_[entity_type];
      break;
    default:
      LOG(ERROR) << "Weight_type is invalid.cur value:" << weight_type;
  }
  return weight;
}

float DGraphShard::_GetWeightSum(WeightType weight_type) {
  float weight = 0.0;
  switch (weight_type) {
    case VERTEX_WEIGHT:
      weight = vertex_weights_;
      break;
    case EDGE_WEIGHT:
      weight = edge_weights_;
      break;
    default:
      LOG(ERROR) << "Weight_type is invalid.cur value:" << weight_type;
  }
  return weight;
}

}  // namespace client
}  // namespace galileo
