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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "common/message.h"
#include "common/packer.h"

using namespace galileo::common;

TEST(PackerTest, TestEntityRequest) {
  uint8_t tmp_types[2] = {0, 1};
  galileo::common::ArraySpec<uint8_t> types;
  types.cnt = 2;
  types.data = tmp_types;

  std::vector<uint32_t> counts;
  for (size_t i = 0; i < 2; ++i) {
    counts.push_back(static_cast<uint32_t>(i + 5));
  }

  size_t capacity =
      types.Capacity() + sizeof(size_t) + sizeof(uint32_t) * counts.size();
  std::string str;
  Packer packer = Packer(&str, capacity);
  packer.Pack(types, counts);
  size_t pack_offset = packer.PackEnd();
  ASSERT_EQ(capacity, pack_offset);

  EntityRequest req;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&req.types_, &req.counts_);

  ASSERT_EQ(types.cnt, req.types_.cnt);

  ASSERT_EQ(counts.size(), req.counts_.cnt);
  for (size_t i = 0; i < req.counts_.cnt; ++i) {
    ASSERT_EQ(counts[i], req.counts_.data[i]);
  }
}

TEST(PackerTest, TestVertexReply) {
  std::vector<std::vector<VertexID>> ids;
  for (size_t i = 0; i < 2; ++i) {
    std::vector<VertexID> tmp;
    for (size_t j = 0; j < 5; j++) {
      tmp.push_back(j);
    }
    ids.push_back(tmp);
  }

  std::string str;
  Packer packer = Packer(&str);
  packer.Pack(ids);
  size_t pack_offset = packer.PackEnd();
  ASSERT_EQ(pack_offset, str.size());

  VertexReply reply;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&reply.ids_);
  ASSERT_EQ(reply.ids_.size(), ids.size());
  for (size_t i = 0; i < reply.ids_.size(); ++i) {
    ASSERT_EQ(reply.ids_[i].cnt, ids[i].size());
    for (size_t j = 0; j < reply.ids_[i].cnt; ++j) {
      ASSERT_EQ(reply.ids_[i].data[j], ids[i][j]);
    }
  }
}

TEST(PackerTest, TestEdgeReply) {
  std::vector<std::vector<EdgeID>> ids;
  for (size_t i = 0; i < 2; ++i) {
    std::vector<EdgeID> tmp;
    for (size_t j = 0; j < 5; j++) {
      tmp.push_back({static_cast<EdgeType>(i), static_cast<int64_t>(i + 1),
                     static_cast<int64_t>(i + 2)});
    }
    ids.push_back(tmp);
  }
  std::string str;
  Packer packer = galileo::common::Packer(&str);
  packer.Pack(ids);
  size_t pack_offset = packer.PackEnd();
  ASSERT_EQ(pack_offset, str.size());

  EdgeReply reply;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&reply.ids_);
  ASSERT_EQ(reply.ids_.size(), ids.size());
  for (size_t i = 0; i < reply.ids_.size(); ++i) {
    ASSERT_EQ(reply.ids_[i].cnt, ids[i].size());
    for (size_t j = 0; j < reply.ids_[i].cnt; ++j) {
      ASSERT_EQ(reply.ids_[i].data[j].src_id, ids[i][j].src_id);
      ASSERT_EQ(reply.ids_[i].data[j].dst_id, ids[i][j].dst_id);
      ASSERT_EQ(reply.ids_[i].data[j].edge_type, ids[i][j].edge_type);
    }
  }
}

TEST(PackerTest, TestNeighborRequest) {
  std::vector<VertexID> ids;
  for (size_t i = 0; i < 5; ++i) {
    ids.push_back(i);
  }
  uint8_t tmp_types[2] = {0, 1};
  galileo::common::ArraySpec<uint8_t> types;
  types.cnt = 2;
  types.data = tmp_types;
  uint32_t count = 5;
  bool need_weight = true;
  size_t len = sizeof(size_t) + sizeof(VertexID) * ids.size() +
               types.Capacity() + sizeof(count) + sizeof(need_weight);
  std::string str;
  Packer packer = Packer(&str, len);
  packer.Pack(ids, types, count, need_weight);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(len, str.size());
  ASSERT_EQ(pack_offset, str.size());

  NeighborRequest req;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&req.ids_, &req.edge_types_, &req.cnt, &req.need_weight_);

  ASSERT_EQ(req.ids_.cnt, ids.size());
  ASSERT_EQ(req.edge_types_.cnt, types.cnt);

  ASSERT_EQ(req.cnt, count);
  ASSERT_EQ(req.need_weight_, need_weight);
}

TEST(PackerTest, TestNeighborReplyWithWeight) {
  IDWeight* tmp_pairs = new IDWeight[2];
  for (size_t i = 0; i < 2; ++i) {
    tmp_pairs[i].id_ = i;
    tmp_pairs[i].weight_ = static_cast<float>(0.1 * static_cast<float>(i + 1));
  }
  galileo::common::ArraySpec<IDWeight> pairs;
  pairs.cnt = 2;
  pairs.data = tmp_pairs;

  std::string str;
  Packer packer = Packer(&str);
  size_t count = 1;
  packer.Pack(count);
  packer.Pack(pairs);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(pack_offset, str.size());

  NeighborReplyWithWeight reply;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&reply.neighbors_);

  ASSERT_EQ(reply.neighbors_.size(), 1);
  for (size_t i = 0; i < reply.neighbors_.size(); ++i) {
    ASSERT_EQ(reply.neighbors_[i].cnt, pairs.cnt);
    for (size_t j = 0; j < reply.neighbors_[i].cnt; ++j) {
      ASSERT_EQ(reply.neighbors_[i].data[j].id_, pairs.data[j].id_);
      ASSERT_EQ(reply.neighbors_[i].data[j].weight_, pairs.data[j].weight_);
    }
  }
}

TEST(PackerTest, TestNeighborReplyWithoutWeight) {
  VertexID tmp_ids[2] = {1, 2};
  galileo::common::ArraySpec<VertexID> pairs;
  pairs.cnt = 2;
  pairs.data = tmp_ids;

  std::string str;
  Packer packer = Packer(&str);
  size_t count = 1;
  packer.Pack(count);
  packer.Pack(pairs);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(pack_offset, str.size());

  NeighborReplyWithoutWeight reply;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&reply.neighbors_);

  ASSERT_EQ(reply.neighbors_.size(), 1);
  for (size_t i = 0; i < reply.neighbors_.size(); ++i) {
    ASSERT_EQ(reply.neighbors_[i].cnt, pairs.cnt);
    for (size_t j = 0; j < reply.neighbors_[i].cnt; ++j) {
      ASSERT_EQ(reply.neighbors_[i].data[j], pairs.data[j]);
    }
  }
}

TEST(PackerTest, TestVertexFeatureRequest) {
  galileo::common::ArraySpec<VertexID> ids;
  VertexID tmp_ids[2] = {11, 22};
  ids.cnt = 2;
  ids.data = tmp_ids;

  std::vector<galileo::common::ArraySpec<char>> features;

  galileo::common::ArraySpec<char> f1;
  f1.cnt = 5;
  f1.data = "price";

  galileo::common::ArraySpec<char> f2;
  f2.cnt = 9;
  f2.data = "click_num";
  galileo::common::ArraySpec<char> f3;
  f3.cnt = 3;
  f3.data = "buy";
  features.push_back(f1);
  features.push_back(f2);
  features.push_back(f3);

  galileo::common::ArraySpec<uint32_t> max_dims;
  uint32_t tmp_dims[3] = {1, 2, 3};
  max_dims.cnt = 3;
  max_dims.data = tmp_dims;

  std::string str;
  Packer packer = Packer(&str);
  packer.Pack(ids, features, max_dims);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(pack_offset, str.size());

  VertexFeatureRequest req;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&req.ids_, &req.features_, &req.max_dims_);

  ASSERT_EQ(req.ids_.cnt, ids.cnt);
  for (size_t i = 0; i < req.ids_.cnt; ++i) {
    ASSERT_EQ(req.ids_.data[i], ids.data[i]);
  }

  ASSERT_EQ(req.features_.size(), features.size());
  for (size_t i = 0; i < req.features_.size(); ++i) {
    ASSERT_EQ(req.features_[i].cnt, features[i].cnt);
  }

  ASSERT_EQ(req.max_dims_.cnt, max_dims.cnt);
  for (size_t i = 0; i < req.max_dims_.cnt; ++i) {
    ASSERT_EQ(req.max_dims_.data[i], max_dims.data[i]);
  }
}

TEST(PackerTest, TestEdgeFeatureRequest) {
  galileo::common::ArraySpec<EdgeID> ids;
  EdgeID* tmp_ids = new EdgeID[2];
  for (size_t i = 0; i < 2; ++i) {
    tmp_ids[i].src_id = static_cast<int64_t>(i + 1);
    tmp_ids[i].dst_id = static_cast<int64_t>(i + 2);
    tmp_ids[i].edge_type = static_cast<uint8_t>(i);
  }
  ids.cnt = 2;
  ids.data = tmp_ids;

  std::vector<galileo::common::ArraySpec<char>> features;

  galileo::common::ArraySpec<char> f1;
  f1.cnt = 5;
  f1.data = "price";

  galileo::common::ArraySpec<char> f2;
  f2.cnt = 9;
  f2.data = "click_num";
  galileo::common::ArraySpec<char> f3;
  f3.cnt = 3;
  f3.data = "buy";
  features.push_back(f1);
  features.push_back(f2);
  features.push_back(f3);

  galileo::common::ArraySpec<uint32_t> max_dims;
  uint32_t tmp_dims[3] = {1, 2, 3};
  max_dims.cnt = 3;
  max_dims.data = tmp_dims;

  std::string str;
  Packer packer = Packer(&str);
  packer.Pack(ids, features, max_dims);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(pack_offset, str.size());

  EdgeFeatureRequest req;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&req.ids_, &req.features_, &req.max_dims_);

  ASSERT_EQ(req.ids_.cnt, ids.cnt);
  for (size_t i = 0; i < req.ids_.cnt; ++i) {
    ASSERT_EQ(req.ids_.data[i].src_id, ids.data[i].src_id);
    ASSERT_EQ(req.ids_.data[i].dst_id, ids.data[i].dst_id);
    ASSERT_EQ(req.ids_.data[i].edge_type, ids.data[i].edge_type);
  }

  ASSERT_EQ(req.features_.size(), features.size());
  for (size_t i = 0; i < req.features_.size(); ++i) {
    ASSERT_EQ(req.features_[i].cnt, features[i].cnt);
  }

  ASSERT_EQ(req.max_dims_.cnt, max_dims.cnt);
  for (size_t i = 0; i < req.max_dims_.cnt; ++i) {
    ASSERT_EQ(req.max_dims_.data[i], max_dims.data[i]);
  }
}

TEST(PackerTest, TestFeatureReply) {
  galileo::common::ArraySpec<galileo::proto::DataType> f_types;
  galileo::proto::DataType tmp_ftype[3] = {galileo::proto::DT_INT32,
                                           galileo::proto::DT_UINT64,
                                           galileo::proto::DT_FLOAT};
  f_types.cnt = 3;
  f_types.data = tmp_ftype;
  std::vector<std::vector<galileo::common::ArraySpec<char>>> features;
  size_t f1 = 1;
  char* val1 = (char*)&f1;
  galileo::common::ArraySpec<char> feat1;
  feat1.cnt = sizeof(f1);
  feat1.data = val1;
  std::vector<galileo::common::ArraySpec<char>> features1;
  features1.push_back(feat1);

  float f2 = 2.0;
  char* val2 = (char*)&f2;
  galileo::common::ArraySpec<char> feat2;
  feat2.cnt = sizeof(f2);
  feat2.data = val2;
  std::vector<galileo::common::ArraySpec<char>> features2;
  features2.push_back(feat2);

  char f3[] = "f3_value";
  galileo::common::ArraySpec<char> feat3;
  feat3.cnt = 8;
  feat3.data = f3;
  std::vector<galileo::common::ArraySpec<char>> features3;
  features3.push_back(feat3);

  features.push_back(features1);
  features.push_back(features2);
  features.push_back(features3);

  std::string str;
  Packer packer = Packer(&str);
  packer.Pack(features, f_types);
  size_t pack_offset = packer.PackEnd();

  ASSERT_EQ(pack_offset, str.size());

  FeatureReply reply;
  Packer unpacker = Packer(&str);
  unpacker.UnPack(&reply.features_, &reply.features_type_);

  ASSERT_EQ(reply.features_.size(), features.size());
  for (size_t i = 0; i < reply.features_.size(); ++i) {
    ASSERT_EQ(reply.features_[i].size(), features[i].size());
    for (size_t j = 0; j < reply.features_[i].size(); ++j) {
      ASSERT_EQ(reply.features_[i][j].cnt, features[i][j].cnt);
    }
  }

  ASSERT_EQ(reply.features_type_.cnt, f_types.cnt);
  for (size_t i = 0; i < reply.features_type_.cnt; ++i) {
    ASSERT_EQ(reply.features_type_.data[i], f_types.data[i]);
  }
}
