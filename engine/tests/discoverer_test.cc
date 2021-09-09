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

#include "discovery/discoverer.h"
#include "discovery/register.h"

#include <gtest/gtest.h>

namespace galileo {
namespace discovery {

class DiscoveryTest : public testing::Test {
 public:
  DiscoveryTest()
      : register_("127.0.0.1:2181", "/galileo_discovery_test"),
        discovery_("127.0.0.1:2181", "/galileo_discovery_test"),
        shard_callbacks({[](const std::string &addr) {
                           std::cerr << "cb online " << addr << std::endl;
                         },
                         [](const std::string &addr) {
                           std::cerr << "cb offline " << addr << std::endl;
                         }}) {}

  Register register_;
  Discoverer discovery_;
  ShardCallbacks shard_callbacks;

  void SetUp() override {
    register_.AddShard({0, "127.0.0.1:1234"},
                       {2, 2, 10, 20, {3., 7.}, {2., 8.}});
    register_.AddShard({1, "127.0.0.1:5678"},
                       {2, 2, 15, 30, {1., 5.}, {4., 6.}});
    discovery_.SetShardCallbacks(0, &shard_callbacks);
  }
  void TearDown() override {
    discovery_.UnsetShardCallbacks(0, &shard_callbacks);
    register_.RemoveShard({0, "127.0.0.1:1234"});
    register_.RemoveShard({1, "127.0.0.1:5678"});
  }
};

TEST_F(DiscoveryTest, Test1) {
  ASSERT_EQ(2, discovery_.GetShardsNum());
  ASSERT_EQ(2, discovery_.GetPartitionsNum());
  ASSERT_EQ(25, discovery_.GetVertexSize());
  ASSERT_EQ(50, discovery_.GetEdgeSize());
  std::vector<float> vw, ew;
  discovery_.GetVertexWeightSum(0, &vw);
  ASSERT_EQ(3., vw[0]);
  ASSERT_EQ(7., vw[1]);
  discovery_.GetEdgeWeightSum(1, &ew);
  ASSERT_EQ(4., ew[0]);
  ASSERT_EQ(6., ew[1]);
}

}  // namespace discovery
}  // namespace galileo
