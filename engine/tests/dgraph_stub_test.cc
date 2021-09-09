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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "client/dgraph_config.h"
#include "client/dgraph_creator.h"
#include "client/dgraph_impl.h"
#include "client/dgraph_stub.h"
#include "client/dgraph_type.h"

#include "gtest/gtest.h"

namespace galileo {
namespace client {

std::unique_ptr<DGraphStub> NewDGraphStub(DGraphConfig &config) {
  DGraphStub *graph = new DGraphStub();
  if (!graph->Initialize(config)) {
    delete graph;
    graph = nullptr;
  }
  return std::unique_ptr<DGraphStub>(graph);
}

void InitConfig(DGraphConfig *config) {
  std::string zk_server = "1.0.0.0:8888";
  std::string zk_path = "/dgraph/cluster";
  config->Add("zk_server", zk_server);
  config->Add("zk_path", zk_path);
}

TEST(DGraphStubTest, TestCollectVertex) {
  DGraphConfig config;
  InitConfig(&config);
  size_t shard_num = 2;
  auto graph_impl = NewDGraphStub(config);

  galileo::common::ArraySpec<uint8_t> types;
  types.cnt = 2;
  uint8_t tmp_types[2] = {0, 1};
  types.data = tmp_types;

  auto mycallback = [](bool status, CollectVertexRes &vertex_res) {
    ASSERT_EQ(status, true);
    std::cout << "shard num:" << vertex_res.size();
    for (auto &shard_vertices : vertex_res) {
      std::cout << "vertex num:" << shard_vertices.cnt << std::endl;
      for (size_t i = 0; i < shard_vertices.cnt; ++i) {
        std::cout << "vertex id:" << shard_vertices.data[i] << std::endl;
      }
    }
  };
  graph_impl->CollectEntity<galileo::common::VertexReply, CollectVertexRes>(
      galileo::common::SAMPLE_VERTEX, types, 5, mycallback);
}

TEST(DGraphStubTest, TestCollectEdge) {
  DGraphConfig config;
  InitConfig(&config);
  size_t shard_num = 2;
  auto graph_impl = NewDGraphStub(config);

  galileo::common::ArraySpec<uint8_t> types;
  types.cnt = 2;
  uint8_t tmp_types[2] = {0, 1};
  types.data = tmp_types;

  auto mycallback = [](bool status, CollectEdgeRes &res) {
    ASSERT_EQ(status, true);
    std::cout << "shard num:" << res.size();
    for (auto &shard_edges : res) {
      std::cout << "edge num:" << shard_edges.cnt << std::endl;
      for (size_t i = 0; i < shard_edges.cnt; ++i) {
        std::cout << "edge src_id:" << shard_edges.data[i].src_id << std::endl;
        std::cout << "edge dst_id:" << shard_edges.data[i].dst_id << std::endl;
        std::cout << "edge edge_type:" << shard_edges.data[i].edge_type
                  << std::endl;
      }
    }
  };
  graph_impl->CollectEntity<galileo::common::EdgeReply, CollectEdgeRes>(
      galileo::common::SAMPLE_EDGE, types, 5, mycallback);
}

}  // namespace client
}  // namespace galileo
