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
#include <string>
#include <vector>

#include "common/types.h"

#include "client/rpc_client.h"
#include "discovery/discoverer.h"

namespace galileo {
namespace client {

enum WeightType {
  VERTEX_WEIGHT,
  EDGE_WEIGHT,
};

class DGraphShard {
 public:
  DGraphShard();

  float GetWeight(WeightType weight_type, int entity_type);

  bool Initilize(uint32_t id,
                 std::shared_ptr<galileo::discovery::Discoverer> discoverer,
                 const DGraphConfig &config);

  void Collect(
      const galileo::proto::QueryRequest &rpc_request,
      galileo::proto::QueryResponse *rpc_response,
      std::function<void(bool is_ok, uint32_t shard_id, std::string *response)>
          callback) const;

 private:
  float _GetWeightSum(WeightType weight_type);

 private:
  std::unique_ptr<galileo::rpc::Client> rpc_client_;
  uint32_t id_;
  float vertex_weights_;
  float edge_weights_;
  std::vector<float> vertex_type_weight_;
  std::vector<float> edge_type_weight_;
};

}  // namespace client
}  // namespace galileo
