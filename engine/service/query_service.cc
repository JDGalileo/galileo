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

#include "service/query_service.h"

#include <brpc/controller.h>
#include <bthread/bthread.h>
#include <glog/logging.h>

#include "common/message.h"
#include "common/packer.h"
#include "service/graph.h"

namespace galileo {
namespace service {

struct AsyncJob {
  std::shared_ptr<Graph> graph;
  const galileo::proto::QueryRequest* request;
  galileo::proto::QueryResponse* response;
  google::protobuf::Closure* done;
  void Run();
  void RunAndDelete() {
    Run();
    delete this;
  }
};

#define UNPACK_WITH_CHECK(DATA)              \
  {                                          \
    bool ret = request_packer.UnPack(&DATA); \
    assert(ret);                             \
    if (!ret) {                              \
      LOG(ERROR) << " UnPack error";         \
      return;                                \
    }                                        \
  }

void AsyncJob::Run() {
  brpc::ClosureGuard done_guard(done);
  if (graph == nullptr) {
    LOG(ERROR) << " Call init first";
    return;
  }
  std::string pack_content;
  galileo::common::Packer response_packer(&pack_content);
  galileo::common::OperatorType op_type =
      (galileo::common::OperatorType)(request->op_type());
  galileo::common::Packer request_packer(
      const_cast<galileo::proto::QueryRequest*>(request)->mutable_data());
  bool op_ret = false;
  switch (op_type) {
    case galileo::common::SAMPLE_VERTEX: {
      galileo::common::EntityRequest entity_request;
      UNPACK_WITH_CHECK(entity_request.types_);
      UNPACK_WITH_CHECK(entity_request.counts_);
      op_ret = graph->SampleVertex(entity_request, &response_packer);
    } break;
    case galileo::common::SAMPLE_EDGE: {
      galileo::common::EntityRequest entity_request;
      UNPACK_WITH_CHECK(entity_request.types_);
      UNPACK_WITH_CHECK(entity_request.counts_);
      op_ret = graph->SampleEdge(entity_request, &response_packer);
    } break;
    case galileo::common::SAMPLE_NEIGHBOR:
    case galileo::common::GET_TOPK_NEIGHBOR:
    case galileo::common::GET_NEIGHBOR: {
      galileo::common::NeighborRequest neighbor_request;
      UNPACK_WITH_CHECK(neighbor_request.ids_);
      UNPACK_WITH_CHECK(neighbor_request.edge_types_);
      UNPACK_WITH_CHECK(neighbor_request.cnt);
      UNPACK_WITH_CHECK(neighbor_request.need_weight_);
      op_ret =
          graph->QueryNeighbors(op_type, neighbor_request, &response_packer);
    } break;
    case galileo::common::GET_VERTEX_FEATURE: {
      galileo::common::VertexFeatureRequest vertex_feature_request;
      UNPACK_WITH_CHECK(vertex_feature_request.ids_);
      UNPACK_WITH_CHECK(vertex_feature_request.features_);
      UNPACK_WITH_CHECK(vertex_feature_request.max_dims_);
      op_ret =
          graph->GetVertexFeature(vertex_feature_request, &response_packer);
    } break;
    case galileo::common::GET_EDGE_FEATURE: {
      galileo::common::EdgeFeatureRequest edge_feature_request;
      UNPACK_WITH_CHECK(edge_feature_request.ids_);
      UNPACK_WITH_CHECK(edge_feature_request.features_);
      UNPACK_WITH_CHECK(edge_feature_request.max_dims_);
      op_ret = graph->GetEdgeFeature(edge_feature_request, &response_packer);
    } break;
    default:
      op_ret = false;
      LOG(ERROR) << " Operator type is not support.op:" << op_type;
      break;
  }
  if (op_ret) {
    response_packer.PackEnd();
  } else {
    pack_content.clear();
  }
  response->set_data(std::move(pack_content));
}

static void* process_thread(void* args) {
  AsyncJob* job = static_cast<AsyncJob*>(args);
  job->RunAndDelete();
  return NULL;
}

bool QueryService::Init(const galileo::service::GraphConfig& config) {
  if (graph_) {
    LOG(ERROR) << " Aleady init query service";
    return false;
  }
  graph_ = std::make_shared<Graph>();
  return graph_->Init(config);
}

const galileo::common::ShardMeta QueryService::QueryShardMeta() {
  return graph_->QueryShardMeta();
}

void QueryService::Query(::google::protobuf::RpcController* cntl,
                         const galileo::proto::QueryRequest* request,
                         galileo::proto::QueryResponse* response,
                         ::google::protobuf::Closure* done) {
  if (!cntl || !done) {
    LOG(ERROR) << " Controller or done is null";
    return;
  }
  brpc::ClosureGuard done_guard(done);
  AsyncJob* job = new AsyncJob;
  job->graph = graph_;
  job->request = request;
  job->response = response;
  job->done = done;
  bthread_t th;
  int res = bthread_start_background(&th, NULL, process_thread, job);
  if (res != 0) {
    LOG(ERROR) << "Start backgroud bthread fail.res:" << res;
  }
  done_guard.release();
}

}  // namespace service
}  // namespace galileo
