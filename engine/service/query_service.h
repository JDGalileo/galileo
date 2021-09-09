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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "proto/rpc.pb.h"
#include "service/config.h"

namespace galileo {
namespace service {

class Graph;

class QueryService : public galileo::proto::GraphQueryService {
 public:
  QueryService() {}
  virtual ~QueryService() {}

 public:
  bool Init(const galileo::service::GraphConfig& config);

  const galileo::common::ShardMeta QueryShardMeta();

  void Query(::google::protobuf::RpcController* cntl,
             const galileo::proto::QueryRequest* request,
             galileo::proto::QueryResponse* response,
             ::google::protobuf::Closure* done);

 private:
  std::shared_ptr<Graph> graph_;
};

}  // namespace service
}  // namespace galileo
