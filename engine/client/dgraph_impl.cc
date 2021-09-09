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

#include "client/dgraph_impl.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <mutex>

#include "common/message.h"
#include "common/sampler.h"
#include "glog/logging.h"

namespace galileo {
namespace client {

using EdgeArraySpec = galileo::common::EdgeArraySpec;

class Notifier {
 public:
  Notifier() : notified_(false) {}
  ~Notifier() {}

  void Notify() {
    std::unique_lock<std::mutex> lk(mutex_);
    assert(!HasBeenNotified());
    notified_ = true;
    cv_.notify_one();
  }

  bool HasBeenNotified() const { return notified_; }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [&] { return HasBeenNotified(); });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool notified_;
};

DGraphImpl::DGraphImpl() : graph_stub_(new DGraphStub()) {}

DGraphImpl::~DGraphImpl() {}

bool DGraphImpl::Initialize(const DGraphConfig &config) {
  return graph_stub_->Initialize(config);
}

int DGraphImpl::CollectEntity(const std::string &category,
                              const ArraySpec<uint8_t> &types, uint32_t count,
                              ITensorAlloc *alloc) const {
  if (0 == count) {
    LOG(ERROR) << " The count param is invalid."
               << " count:" << count;
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr.";
    return -1;
  }
  Notifier notifier;
  size_t expected_cnt = static_cast<size_t>(count);
  int tensor_num = 0;
  if ("vertex" == category) {
    auto my_callback = [this, expected_cnt, alloc, &tensor_num, &notifier,
                        &types](bool status, const CollectVertexRes &res) {
      if (status) {
        std::vector<size_t> valid_types_idx;
        size_t type_num = 0 == types.cnt ? 1 : types.cnt;
        VertexID *data = (VertexID *)alloc->AllocListTensor(
            CT_INT64, {static_cast<long long>(type_num),
                       static_cast<long long>(expected_cnt)});
        ++tensor_num;
        if (nullptr == data) {
          LOG(ERROR) << " Alloc vertex tensor memory fail.";
          tensor_num = -1;
          notifier.Notify();
          return;
        }
        size_t buf_idx = 0;
        for (size_t type_idx = 0; type_idx < type_num; ++type_idx) {
          size_t real_cnt = 0;
          for (auto &shard_vertices : res) {
            real_cnt += shard_vertices[type_idx].cnt;
            for (size_t j = 0; j < shard_vertices[type_idx].cnt; ++j) {
              data[buf_idx++] = shard_vertices[type_idx].data[j];
            }
          }
          if (real_cnt != expected_cnt && types.cnt != 0) {
            LOG(ERROR)
                << " The id num of sample vertex is not equal to expected num."
                << " edge type:" << std::to_string(types.data[type_idx])
                << " ,real num:" << real_cnt
                << " expected_num:" << expected_cnt;
            tensor_num = -1;
            break;
          } else if (real_cnt != expected_cnt && 0 == types.cnt) {
            LOG(ERROR) << " The id num of global sample vertex is not equal to "
                          "expected num."
                       << " real num:" << real_cnt
                       << " ,expected_num:" << expected_cnt;
            tensor_num = -1;
            break;
          }
        }
      }
      notifier.Notify();
    };
    graph_stub_->CollectEntity<galileo::common::VertexReply, CollectVertexRes>(
        galileo::common::SAMPLE_VERTEX, types, count, my_callback);
    notifier.WaitForNotification();
  } else if ("edge" == category) {
    auto my_callback = [this, expected_cnt, alloc, &tensor_num, &notifier,
                        &types](bool status, const CollectEdgeRes &res) {
      if (status) {
        std::vector<size_t> valid_types_idx;
        size_t type_num = 0 == types.cnt ? 1 : types.cnt;
        VertexID *srcs = (VertexID *)alloc->AllocListTensor(
            CT_INT64, {static_cast<long long>(type_num),
                       static_cast<long long>(expected_cnt)});
        ++tensor_num;
        if (nullptr == srcs) {
          LOG(ERROR) << " Alloc edges src tensor memory fail.";
          tensor_num = -1;
          notifier.Notify();
          return;
        }
        VertexID *dsts = (VertexID *)alloc->AllocListTensor(
            CT_INT64, {static_cast<long long>(type_num),
                       static_cast<long long>(expected_cnt)});
        ++tensor_num;
        if (nullptr == dsts) {
          LOG(ERROR) << " Alloc edges dst tensor memory fail.";
          tensor_num = -1;
          notifier.Notify();
          return;
        }
        uint8_t *etypes = (uint8_t *)alloc->AllocListTensor(
            CT_UINT8, {static_cast<long long>(type_num),
                       static_cast<long long>(expected_cnt)});
        ++tensor_num;
        if (nullptr == etypes) {
          LOG(ERROR) << " Alloc edges etype tensor memory fail.";
          tensor_num = -1;
          notifier.Notify();
          return;
        }
        size_t buf_idx = 0;
        for (size_t type_idx = 0; type_idx < type_num; ++type_idx) {
          size_t real_cnt = 0;
          for (auto &shard_edges : res) {
            real_cnt += shard_edges[type_idx].cnt;
            for (size_t j = 0; j < shard_edges[type_idx].cnt; ++j) {
              srcs[buf_idx] = shard_edges[type_idx].data[j].src_id;
              dsts[buf_idx] = shard_edges[type_idx].data[j].dst_id;
              etypes[buf_idx] = shard_edges[type_idx].data[j].edge_type;
              ++buf_idx;
            }
          }
          if (real_cnt != expected_cnt && types.cnt != 0) {
            LOG(ERROR)
                << " The id num of sample edge is not equal to expected num."
                << " edge type:" << std::to_string(types.data[type_idx])
                << " ,real num:" << real_cnt
                << " ,expected_num:" << expected_cnt;
            tensor_num = -1;
            break;
          } else if (real_cnt != expected_cnt && 0 == types.cnt) {
            LOG(ERROR) << " The id num of global sample edge is not equal to "
                          "expected num."
                       << " real num:" << real_cnt
                       << " ,expected_num:" << expected_cnt;
            tensor_num = -1;
            break;
          }
        }
      }
      notifier.Notify();
    };
    graph_stub_->CollectEntity<galileo::common::EdgeReply, CollectEdgeRes>(
        galileo::common::SAMPLE_EDGE, types, count, my_callback);
    notifier.WaitForNotification();
  } else {
    LOG(ERROR) << " The category param is invalid. category:" << category;
    return -1;
  }
  return tensor_num;
}

int DGraphImpl::CollectNeighbor(const std::string &category,
                                const ArraySpec<VertexID> &ids,
                                const ArraySpec<uint8_t> &edge_types,
                                uint32_t count, bool need_weight,
                                ITensorAlloc *alloc) const {
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr .";
    return -1;
  }
  int tensor_num = 0;
  if ("sample" == category) {
    if (need_weight) {
      tensor_num = this->_GetLimitedNeighborWithWeight(
          galileo::common::SAMPLE_NEIGHBOR, ids, edge_types, count, alloc);
    } else {
      tensor_num = this->_GetLimitedNeighborWithoutWeight(
          galileo::common::SAMPLE_NEIGHBOR, ids, edge_types, count, alloc);
    }
  } else if ("topk" == category) {
    if (need_weight) {
      tensor_num = this->_GetLimitedNeighborWithWeight(
          galileo::common::GET_TOPK_NEIGHBOR, ids, edge_types, count, alloc);
    } else {
      tensor_num = this->_GetLimitedNeighborWithoutWeight(
          galileo::common::GET_TOPK_NEIGHBOR, ids, edge_types, count, alloc);
    }
  } else if ("full" == category) {
    if (need_weight) {
      tensor_num =
          this->_GetFullNeighborWithWeight(ids, edge_types, count, alloc);

    } else {
      tensor_num =
          this->_GetFullNeighborWithoutWeight(ids, edge_types, count, alloc);
    }
  } else {
    LOG(ERROR) << " The category param is invalid. category:" << category;
    return -1;
  }
  return tensor_num;
}

int DGraphImpl::CollectFeature(const std::string &category, const char *ids,
                               const std::vector<ArraySpec<char>> &features,
                               const ArraySpec<uint32_t> &max_dims,
                               ITensorAlloc *alloc) const {
  if (nullptr == ids) {
    LOG(ERROR) << " The ids param is nullptr.";
    return -1;
  }
  if (0 == features.size()) {
    LOG(ERROR) << " The features param is empty.";
    return -1;
  }
  if (max_dims.IsEmpty()) {
    LOG(ERROR) << " The max_dims param is empty.";
    return -1;
  }
  if (max_dims.cnt != features.size()) {
    LOG(ERROR) << " The max_dims count is not equal to features count."
               << " max_dim_count:" << max_dims.cnt
               << " ,features size:" << features.size();
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr .";
    return -1;
  }
  int tensor_num = 0;
  if ("vertex" == category) {
    tensor_num = this->_GetFeature(galileo::common::GET_VERTEX_FEATURE, ids,
                                   features, max_dims, alloc);
  } else if ("edge" == category) {
    tensor_num = this->_GetFeature(galileo::common::GET_EDGE_FEATURE, ids,
                                   features, max_dims, alloc);
  } else {
    LOG(ERROR) << " The category param is invalid. category:" << category;
    return -1;
  }
  return tensor_num;
}

int DGraphImpl::CollectPodFeature(const std::string &category, const char *ids,
                                  const std::vector<ArraySpec<char>> &features,
                                  const ArraySpec<uint32_t> &max_dims,
                                  ITensorAlloc *alloc) const {
  if (nullptr == ids) {
    LOG(ERROR) << " The ids param is nullptr.";
    return -1;
  }
  if (0 == features.size()) {
    LOG(ERROR) << " The features param is empty.";
    return -1;
  }
  if (max_dims.IsEmpty()) {
    LOG(ERROR) << "The max_dims param is empty.";
    return -1;
  }
  if (max_dims.cnt != features.size()) {
    LOG(ERROR) << " The max_dims count is not equal to features count."
               << " max_dim_count:" << max_dims.cnt
               << " ,features size:" << features.size();
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr.";
    return -1;
  }
  int tensor_num = 0;
  if ("vertex" == category) {
    tensor_num = this->_GetPodFeature(galileo::common::GET_VERTEX_FEATURE, ids,
                                      features, max_dims, alloc);
  } else if ("edge" == category) {
    tensor_num = this->_GetPodFeature(galileo::common::GET_EDGE_FEATURE, ids,
                                      features, max_dims, alloc);
  } else {
    LOG(ERROR) << " The op param is invalid. op:" << category;
    return false;
  }
  return tensor_num;
}

int DGraphImpl::CollectSeqByMultiHop(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    const ArraySpec<uint32_t> &counts, bool need_weight,
    ITensorAlloc *alloc) const {
  if (counts.IsEmpty()) {
    LOG(ERROR) << " The counts param is empty.";
    return -1;
  }
  if (metapath.size() != counts.cnt) {
    LOG(ERROR) << " The metapath size is not equal to counts size."
               << " metapath size:" << metapath.size()
               << " ,counts size:" << counts.cnt;
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr.";
    return -1;
  }
  int tensor_num = 0;
  if (need_weight) {
    tensor_num =
        this->_SampleSeqWithWeightByMultiHop(ids, metapath, counts, alloc);
  } else {
    tensor_num =
        this->_SampleSeqWithoutWeightByMultiHop(ids, metapath, counts, alloc);
  }
  return tensor_num;
}

int DGraphImpl::CollectSeqByRWWithBias(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
    float p, float q, ITensorAlloc *alloc) const {
  if (0 == metapath.size()) {
    LOG(ERROR) << " The size of metapath param is zero.";
    return -1;
  }
  if (p < 0 || std::fabs(p - 0.0) <= FLT_EPSILON) {
    LOG(ERROR) << " The p param is invalid.p:" << p;
    return -1;
  }
  if (q < 0 || std::fabs(q - 0.0) <= FLT_EPSILON) {
    LOG(ERROR) << " The q param is invalid.q:" << q;
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr .";
    return -1;
  }
  return this->_SampleSeqByRWWithBias(ids, metapath, repetition, p, q, alloc);
}

int DGraphImpl::CollectPairByRWWithBias(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
    float p, float q, uint32_t context_size, ITensorAlloc *alloc) const {
  if (0 == metapath.size()) {
    LOG(ERROR) << " The size of metapath param is zero.";
    return -1;
  }
  if (p < 0 || std::fabs(p - 0.0) <= FLT_EPSILON) {
    LOG(ERROR) << " The p param is invalid.p:" << p;
    return -1;
  }
  if (q < 0 || std::fabs(q - 0.0) <= FLT_EPSILON) {
    LOG(ERROR) << " The q param is invalid.q:" << q;
    return -1;
  }
  if (0 == context_size) {
    LOG(ERROR) << " The context_size param must be more than zero.";
    return -1;
  }
  if (nullptr == alloc) {
    LOG(ERROR) << " The alloc param is nullptr .";
    return -1;
  }
  bool res = false;
  int tensor_num = 0;
  size_t walk_len = metapath.size();
  size_t seq_num_per_vertex = walk_len + 1;
  size_t seq_len = ids.cnt * repetition * seq_num_per_vertex;
  VertexID *rw_sequences = new VertexID[seq_len];
  std::vector<VertexID> padding_ids;
  for (uint32_t i = 0; i < repetition; ++i) {
    for (size_t j = 0; j < ids.cnt; ++j) {
      padding_ids.push_back(ids.data[j]);
    }
  }
  for (size_t i = 0; i < padding_ids.size(); ++i) {
    rw_sequences[i * seq_num_per_vertex] = padding_ids[i];
  }
  size_t pair_num = this->_CalPairNum(ids.cnt, repetition, walk_len,
                                      static_cast<size_t>(context_size));

  VertexID *pairs_buff = (VertexID *)alloc->AllocListTensor(
      CT_INT64, {static_cast<long long>(pair_num), 2});
  ++tensor_num;
  if (nullptr == pairs_buff && pair_num > 0) {
    LOG(ERROR) << " Alloc pairs_buff tensor memory is failed.";
    return -1;
  }
  if (std::fabs(p - 1.0) <= FLT_EPSILON && std::fabs(q - 1.0) <= FLT_EPSILON) {
    res = this->_SampleSeqWithoutBias(padding_ids, metapath, rw_sequences);
  } else {
    res = this->_SampleSeqWithBias(padding_ids, metapath, p, q, rw_sequences);
  }
  if (!res) {
    delete[] rw_sequences;
    return -1;
  }

  this->_BuildRandomWalkSeqPair(rw_sequences,
                                static_cast<int>(padding_ids.size()),
                                static_cast<int>(seq_num_per_vertex),
                                static_cast<int>(context_size), pairs_buff);
  delete[] rw_sequences;
  return tensor_num;
}

bool DGraphImpl::CollectGraphMeta(GraphMeta *meta_info) const {
  if (nullptr == meta_info) {
    LOG(ERROR) << " Meta info is nullptr.";
    return false;
  }
  return graph_stub_->CollectGraphMeta(meta_info);
}

int DGraphImpl::_GetLimitedNeighborWithWeight(
    galileo::common::OperatorType op, const ArraySpec<VertexID> &ids,
    const ArraySpec<uint8_t> &edge_types, uint32_t count,
    ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  size_t expected_cnt = static_cast<size_t>(count);
  auto my_callback = [this, expected_cnt, alloc, &ids, &tensor_num, &notifier](
                         bool status, CollectNeighborResWithWeight &res) {
    if (status) {
      VertexID *neighbors = (VertexID *)alloc->AllocListTensor(
          CT_INT64, {static_cast<long long>(ids.cnt),
                     static_cast<long long>(expected_cnt)});
      ++tensor_num;
      if (nullptr == neighbors && !res.empty()) {
        LOG(ERROR) << " Alloc neighbors tensor memory fail.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      float *weights = (float *)alloc->AllocListTensor(
          CT_FLOAT, {static_cast<long long>(ids.cnt),
                     static_cast<long long>(expected_cnt)});
      ++tensor_num;
      if (nullptr == weights && !res.empty()) {
        LOG(ERROR) << " Alloc weights tensor memory fail.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      size_t idx = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i].cnt != expected_cnt) {
          LOG(ERROR) << " The neighbor num of id is invalid."
                     << " id:" << ids.data[i]
                     << " ,expected num:" << expected_cnt
                     << " ,real num:" << res[i].cnt;
          tensor_num = -1;
          break;
        }
        for (size_t j = 0; j < res[i].cnt; ++j) {
          neighbors[idx] = res[i].data[j].id_;
          weights[idx] = res[i].data[j].weight_;
          ++idx;
        }
      }
    }
    notifier.Notify();
  };
  graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithWeight,
                               CollectNeighborResWithWeight>(
      op, ids, edge_types, count, true, my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

int DGraphImpl::_GetLimitedNeighborWithoutWeight(
    galileo::common::OperatorType op, const ArraySpec<VertexID> &ids,
    const ArraySpec<uint8_t> &edge_types, uint32_t count,
    ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  size_t expected_cnt = static_cast<size_t>(count);
  auto my_callback = [this, expected_cnt, alloc, &ids, &tensor_num, &notifier](
                         bool status, CollectNeighborResWithoutWeight &res) {
    if (status) {
      VertexID *neighbors = (VertexID *)alloc->AllocListTensor(
          CT_INT64, {static_cast<long long>(ids.cnt),
                     static_cast<long long>(expected_cnt)});
      ++tensor_num;
      if (nullptr == neighbors && !res.empty()) {
        LOG(ERROR) << " Alloc neighbors tensor memory fail.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      size_t idx = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i].cnt != expected_cnt) {
          LOG(ERROR) << " The neighbor num of id is invalid."
                     << " id:" << ids.data[i]
                     << " ,expected num:" << expected_cnt
                     << " ,real num:" << res[i].cnt;
          tensor_num = -1;
          break;
        }
        for (size_t j = 0; j < res[i].cnt; ++j) {
          neighbors[idx++] = res[i].data[j];
        }
      }
    }
    notifier.Notify();
  };
  graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithoutWeight,
                               CollectNeighborResWithoutWeight>(
      op, ids, edge_types, count, false, my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

int DGraphImpl::_GetFullNeighborWithWeight(const ArraySpec<VertexID> &ids,
                                           const ArraySpec<uint8_t> &edge_types,
                                           uint32_t count,
                                           ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  auto my_callback = [alloc, &ids, &tensor_num, &notifier](
                         bool status, CollectNeighborResWithWeight &res) {
    if (status) {
      size_t total_count = 0;
      for (auto &neighbor : res) {
        total_count += neighbor.cnt;
      }
      VertexID *neighbors = (VertexID *)alloc->AllocListTensor(
          CT_INT64, {static_cast<long long>(total_count)});
      ++tensor_num;
      if (nullptr == neighbors && !res.empty()) {
        LOG(ERROR) << " Alloc neighbors tensor memory fail. alloc num:"
                   << total_count;
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      float *weights = (float *)alloc->AllocListTensor(
          CT_FLOAT, {static_cast<long long>(total_count)});
      ++tensor_num;
      if (nullptr == weights && !res.empty()) {
        LOG(ERROR) << " Alloc weights tensor memory fail.alloc_num:"
                   << total_count;
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      int *neighbor_idx = (int *)alloc->AllocListTensor(
          CT_INT32, {static_cast<long long>(ids.cnt), 2});
      ++tensor_num;
      if (nullptr == neighbor_idx && !res.empty()) {
        LOG(ERROR) << " Alloc neighbor_idx tensor memory fail.alloc_num:"
                   << total_count;
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      int idx = 0, vertex_idx = 0;
      for (auto &neighbor : res) {
        neighbor_idx[2 * vertex_idx] = idx;
        neighbor_idx[2 * vertex_idx + 1] = static_cast<int>(neighbor.cnt);
        ++vertex_idx;
        for (size_t i = 0; i < neighbor.cnt; ++i) {
          neighbors[idx] = neighbor.data[i].id_;
          weights[idx] = neighbor.data[i].weight_;
          ++idx;
        }
      }
    }
    notifier.Notify();
  };
  graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithWeight,
                               CollectNeighborResWithWeight>(
      galileo::common::GET_NEIGHBOR, ids, edge_types, count, true, my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

int DGraphImpl::_GetFullNeighborWithoutWeight(
    const ArraySpec<VertexID> &ids, const ArraySpec<uint8_t> &edge_types,
    uint32_t count, ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  auto my_callback = [alloc, &ids, &tensor_num, &notifier](
                         bool status, CollectNeighborResWithoutWeight &res) {
    if (status) {
      size_t total_count = 0;
      for (auto &neighbor : res) {
        total_count += neighbor.cnt;
      }
      VertexID *neighbors = (VertexID *)alloc->AllocListTensor(
          CT_INT64, {static_cast<long long>(total_count)});
      ++tensor_num;
      if (nullptr == neighbors && !res.empty()) {
        LOG(ERROR) << " Alloc neighbors tensor memory fail.alloc num:"
                   << total_count;
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      int *neighbor_idx = (int *)alloc->AllocListTensor(
          CT_INT32, {static_cast<long long>(ids.cnt), 2});
      ++tensor_num;
      if (nullptr == neighbor_idx && !res.empty()) {
        LOG(ERROR) << " Alloc neighbor_idx tensor memory fail.alloc num:"
                   << total_count;
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      int idx = 0, vertex_idx = 0;
      for (auto &neighbor : res) {
        neighbor_idx[2 * vertex_idx] = idx;
        neighbor_idx[2 * vertex_idx + 1] = static_cast<int>(neighbor.cnt);
        ++vertex_idx;
        for (size_t i = 0; i < neighbor.cnt; ++i) {
          neighbors[idx++] = neighbor.data[i];
        }
      }
    }
    notifier.Notify();
  };
  graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithoutWeight,
                               CollectNeighborResWithoutWeight>(
      galileo::common::GET_NEIGHBOR, ids, edge_types, count, false,
      my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

size_t DGraphImpl::_GetIDCnt(galileo::common::OperatorType op,
                             const char *ids) const {
  if (galileo::common::GET_VERTEX_FEATURE == op) {
    const ArraySpec<VertexID> *vertices = (const ArraySpec<VertexID> *)ids;
    return vertices->cnt;
  } else {
    const EdgeArraySpec *edges = (const EdgeArraySpec *)ids;
    return edges->srcs.cnt;
  }
}

int DGraphImpl::_GetPodFeature(
    galileo::common::OperatorType op, const char *ids,
    const std::vector<ArraySpec<char>> &features_name,
    const ArraySpec<uint32_t> &max_dims, ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  auto my_callback = [this, op, ids, alloc, &features_name, &tensor_num,
                      &notifier](bool status, CollectFeatureRes &res) {
    if (status) {
      int *features_type = nullptr;
      size_t id_cnt = this->_GetIDCnt(op, ids);
      size_t type_count = res.features_type_.cnt;
      if (0 == type_count && id_cnt > 0) {
        LOG(ERROR) << " All of id or feature name is invalid.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      features_type = (int *)alloc->AllocTypesTensor(type_count);
      if (nullptr == features_type && type_count > 0) {
        LOG(ERROR) << " Alloc features type tensor memory fail.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      for (size_t i = 0; i < type_count; ++i) {
        ClientType ftype = TransformDType2CType(res.features_type_.data[i]);
        if (CT_INVALID_TYPE == ftype) {
          LOG(ERROR) << " Get feature type fail."
                     << " client ftype:" << ftype
                     << " ,reply ftype:" << res.features_type_.data[i];
          tensor_num = -1;
          break;
        }
        if (CT_STRING == ftype) {
          LOG(ERROR) << " Get pod feature not support get string type feature.";
          tensor_num = -1;
          break;
        }

        features_type[i] = alloc->GetTensorType(ftype);
        size_t type_size = GetClientTypeCapacity(ftype);
        if (0 == type_size) {
          LOG(ERROR) << " Get type capacity is zero."
                     << " reply ftype:" << res.features_type_.data[i]
                     << " ,client ftype:" << ftype;
          tensor_num = -1;
          break;
        }
        auto &tmp_features = res.features_.at(0);
        char *my_buffer = nullptr;
        size_t expected_dim = 0;
        size_t offset = 0;
        for (size_t j = 0; j < res.features_.size(); ++j) {
          auto &cur_features = res.features_[j];
          if (0 == cur_features.size()) {
            LOG(ERROR) << " Can not find the id."
                       << " id:" << IDToStr(op, ids, j);
            tensor_num = -1;
            notifier.Notify();
            return;
          }
          if (0 == cur_features[i].cnt) {
            std::string tmp_name(features_name[i].data, features_name[i].cnt);
            LOG(ERROR)
                << " Can not find the feature value or can not find the id."
                << " id:" << IDToStr(op, ids, j)
                << " ,feature name:" << tmp_name;
            tensor_num = -1;
            notifier.Notify();
            return;
          }
          if (nullptr == my_buffer) {
            expected_dim = tmp_features[i].cnt / type_size;
            my_buffer = alloc->AllocListTensor(
                ftype, {static_cast<long long>(id_cnt),
                        static_cast<long long>(expected_dim)});
            ++tensor_num;
            if (nullptr == my_buffer) {
              LOG(ERROR) << " Alloc features value tensor memory fail.";
              tensor_num = -1;
              notifier.Notify();
              return;
            }
          }
          size_t cur_dim = cur_features[i].cnt / type_size;
          if (expected_dim != cur_dim) {
            std::string tmp_name(features_name[i].data, features_name[i].cnt);
            LOG(ERROR) << " The features dim is not equal."
                       << " feature name:" << tmp_name
                       << " ,id1:" << IDToStr(op, ids, 0)
                       << " ,dim1:" << expected_dim
                       << " ,id2:" << IDToStr(op, ids, j)
                       << " ,dim2:" << cur_dim;
            tensor_num = -1;
            notifier.Notify();
            return;
          }
          for (size_t k = 0; k < cur_features[i].cnt; ++k) {
            my_buffer[offset++] = cur_features[i].data[k];
          }
        }
      }
    }
    notifier.Notify();
  };
  graph_stub_->CollectFeature(op, ids, features_name, max_dims, my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

int DGraphImpl::_GetFeature(galileo::common::OperatorType op, const char *ids,
                            const std::vector<ArraySpec<char>> &features_name,
                            const ArraySpec<uint32_t> &max_dims,
                            ITensorAlloc *alloc) const {
  int tensor_num = 0;
  Notifier notifier;
  auto my_callback = [this, op, ids, alloc, &features_name, &tensor_num,
                      &notifier](bool status, CollectFeatureRes &res) {
    if (status) {
      size_t id_cnt = this->_GetIDCnt(op, ids);
      size_t type_count = res.features_type_.cnt;
      if (0 == type_count && id_cnt > 0) {
        LOG(ERROR) << " All of id or feature name is invalid.";
        tensor_num = -1;
        notifier.Notify();
        return;
      }
      if (0 == type_count && 0 == id_cnt) {
        // accept empty tensor
        // alloc tensor outside of client
        tensor_num = static_cast<int>(features_name.size());
        notifier.Notify();
        return;
      }
      for (size_t i = 0; i < type_count; ++i) {
        ClientType ftype = TransformDType2CType(res.features_type_.data[i]);
        if (CT_INVALID_TYPE == ftype) {
          LOG(ERROR) << " Get feature type fail."
                     << " client ftype:" << ftype
                     << " ,reply ftype:" << res.features_type_.data[i];
          tensor_num = -1;
          break;
        }
        if (CT_STRING == ftype) {
          char *my_buffer = alloc->AllocListTensor(
              ftype, {static_cast<long long>(id_cnt), 1});
          ++tensor_num;
          if (nullptr == my_buffer) {
            LOG(ERROR) << " Alloc str features value tensor memory fail.";
            tensor_num = -1;
            break;
          }
          size_t idx = 0;
          for (size_t j = 0; j < res.features_.size(); ++j) {
            auto &cur_features = res.features_[j];
            if (0 == cur_features.size()) {
              LOG(ERROR) << " Can not find the id."
                         << "id:" << IDToStr(op, ids, j);
              tensor_num = -1;
              notifier.Notify();
              return;
            }
            if (0 == cur_features[i].cnt) {
              std::string tmp_name(features_name[i].data, features_name[i].cnt);
              LOG(ERROR) << " Feature name is not find or cur id is invalid."
                         << " id:" << IDToStr(op, ids, j)
                         << " ,feature name:" << tmp_name;
              tensor_num = -1;
              notifier.Notify();
              return;
            }
            alloc->FillStringTensor(my_buffer, idx, cur_features[i]);
            ++idx;
          }
        } else {
          size_t type_size = GetClientTypeCapacity(ftype);
          if (0 == type_size) {
            std::string tmp_name(features_name[i].data, features_name[i].cnt);
            LOG(ERROR) << " Get feature type capacity is zero."
                       << " ctype:" << ftype
                       << " ,dtype:" << res.features_type_.data[i]
                       << " ,feature name:" << tmp_name;
            tensor_num = -1;
            break;
          }
          size_t expected_dim = 0;
          char *my_buffer = nullptr;
          size_t offset = 0;
          for (size_t j = 0; j < res.features_.size(); ++j) {
            auto &cur_features = res.features_[j];
            if (0 == cur_features.size()) {
              LOG(ERROR) << " Can not find the id."
                         << "id:" << IDToStr(op, ids, j);
              tensor_num = -1;
              notifier.Notify();
              return;
            }
            if (0 == cur_features[i].cnt) {
              std::string tmp_name(features_name[i].data, features_name[i].cnt);
              LOG(ERROR) << " Can not find the feature value.we use default "
                            "value to fill it."
                         << " id:" << IDToStr(op, ids, j)
                         << " ,feature name:" << tmp_name;
              tensor_num = -1;
              notifier.Notify();
              return;
            }
            if (nullptr == my_buffer) {
              expected_dim = cur_features[i].cnt / type_size;
              my_buffer = alloc->AllocListTensor(
                  ftype, {static_cast<long long>(id_cnt),
                          static_cast<long long>(expected_dim)});
              ++tensor_num;
              if (nullptr == my_buffer) {
                LOG(ERROR) << " Alloc features value tensor memory fail.";
                tensor_num = -1;
                notifier.Notify();
                return;
              }
            }
            size_t cur_dim = cur_features[i].cnt / type_size;
            if (cur_dim != expected_dim) {
              std::string tmp_name(features_name[i].data, features_name[i].cnt);
              LOG(ERROR) << " The features dim is not equal."
                         << " feature name:" << tmp_name
                         << " ,id1:" << IDToStr(op, ids, 0)
                         << " ,dim1:" << expected_dim
                         << " ,id2:" << IDToStr(op, ids, j)
                         << " ,dim2:" << cur_dim;
              tensor_num = -1;
              notifier.Notify();
              return;
            }
            for (size_t k = 0; k < cur_features[i].cnt; ++k) {
              my_buffer[offset++] = cur_features[i].data[k];
            }
          }
        }
      }
    }
    notifier.Notify();
  };

  graph_stub_->CollectFeature(op, ids, features_name, max_dims, my_callback);
  notifier.WaitForNotification();
  return tensor_num;
}

int DGraphImpl::_SampleSeqWithWeightByMultiHop(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    const ArraySpec<uint32_t> &counts, ITensorAlloc *alloc) const {
  int tensor_num = 0;
  size_t seq_len_per_vertex = this->_GetMultiHopSeqNum(counts);
  VertexID *neighbor_buff = (VertexID *)alloc->AllocListTensor(
      CT_INT64, {static_cast<long long>(ids.cnt),
                 static_cast<long long>(seq_len_per_vertex)});
  ++tensor_num;
  if (nullptr == neighbor_buff && ids.cnt > 0) {
    LOG(ERROR) << " Alloc neighbors tensor memory fail.";
    return -1;
  }
  float *weights_buff = (float *)alloc->AllocListTensor(
      CT_FLOAT, {static_cast<long long>(ids.cnt),
                 static_cast<long long>(seq_len_per_vertex)});
  ++tensor_num;
  if (nullptr == weights_buff && ids.cnt > 0) {
    LOG(ERROR) << " Alloc weight tensor memory fail.";
    return -1;
  }
  for (size_t i = 0; i < ids.cnt; ++i) {
    neighbor_buff[seq_len_per_vertex * i] = ids.data[i];
    weights_buff[seq_len_per_vertex * i] = 1.0;
  }
  std::vector<VertexID> tmp_ids;
  size_t child_num_per_vertex = 1;
  size_t collected_num_per_vertex = 1;
  ArraySpec<VertexID> cur_ids(ids.data, ids.cnt);
  bool res_status = false;
  for (size_t i = 0; i < metapath.size(); ++i) {
    Notifier notifier;
    size_t cur_cnt = static_cast<size_t>(counts.data[i]);
    auto my_callback = [this, i, seq_len_per_vertex, child_num_per_vertex,
                        cur_cnt, neighbor_buff, weights_buff,
                        collected_num_per_vertex, &cur_ids, &notifier, &tmp_ids,
                        &res_status, &metapath](
                           bool status, CollectNeighborResWithWeight &res) {
      res_status = status;
      if (status) {
        tmp_ids.clear();
        size_t vertex_idx = 0;
        size_t tmp_collected_num = collected_num_per_vertex;
        for (size_t j = 0; j < res.size(); ++j) {
          if (res[j].cnt != cur_cnt) {
            LOG(ERROR) << " The neighbor num of sampling id is invalid."
                       << " id:" << cur_ids.data[j]
                       << " ,types:" << TypesToStr(metapath[i])
                       << " ,expected num:" << cur_cnt
                       << " ,real num:" << res[j].cnt;
            res_status = false;
            notifier.Notify();
            return;
          }
          if (vertex_idx != j / child_num_per_vertex) {
            vertex_idx = j / child_num_per_vertex;
            tmp_collected_num = collected_num_per_vertex;
          }
          size_t vertex_offset =
              seq_len_per_vertex * vertex_idx + tmp_collected_num;
          for (size_t k = 0; k < res[j].cnt; ++k) {
            size_t buff_idx = vertex_offset + k;
            neighbor_buff[buff_idx] = res[j].data[k].id_;
            weights_buff[buff_idx] = res[j].data[k].weight_;
            if (i < metapath.size() - 1) {
              tmp_ids.push_back(res[j].data[k].id_);
            }
          }
          tmp_collected_num += res[j].cnt;
        }
        cur_ids.cnt = tmp_ids.size();
        cur_ids.data = tmp_ids.data();
      }
      notifier.Notify();
    };
    graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithWeight,
                                 CollectNeighborResWithWeight>(
        galileo::common::SAMPLE_NEIGHBOR, cur_ids, metapath[i], counts.data[i],
        true, my_callback);
    notifier.WaitForNotification();
    if (!res_status) {
      return -1;
    }
    child_num_per_vertex = child_num_per_vertex * cur_cnt;
    collected_num_per_vertex += child_num_per_vertex;
  }
  return tensor_num;
}

int DGraphImpl::_SampleSeqWithoutWeightByMultiHop(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    const ArraySpec<uint32_t> &counts, ITensorAlloc *alloc) const {
  int tensor_num = 0;
  size_t seq_len_per_vertex = this->_GetMultiHopSeqNum(counts);
  VertexID *neighbor_buff = (VertexID *)alloc->AllocListTensor(
      CT_INT64, {static_cast<long long>(ids.cnt),
                 static_cast<long long>(seq_len_per_vertex)});
  ++tensor_num;
  if (nullptr == neighbor_buff && ids.cnt > 0) {
    LOG(ERROR) << " Alloc neighbor tensor memory fail.";
    return -1;
  }
  for (size_t i = 0; i < ids.cnt; ++i) {
    neighbor_buff[seq_len_per_vertex * i] = ids.data[i];
  }
  bool res_status = false;
  res_status = this->_SampleNeighborWithoutWeightByMultiHop(
      ids, metapath, counts, seq_len_per_vertex, neighbor_buff);
  if (!res_status) {
    return -1;
  }
  return tensor_num;
}

int DGraphImpl::_SampleSeqByRWWithBias(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
    float p, float q, ITensorAlloc *alloc) const {
  std::vector<VertexID> padding_ids;
  for (uint32_t i = 0; i < repetition; ++i) {
    for (size_t j = 0; j < ids.cnt; ++j) {
      padding_ids.push_back(ids.data[j]);
    }
  }
  int tensor_num = 0;
  size_t seq_len = metapath.size() + 1;
  VertexID *seq_buff = (VertexID *)alloc->AllocListTensor(
      CT_INT64, {static_cast<long long>(padding_ids.size()),
                 static_cast<long long>(seq_len)});
  ++tensor_num;
  if (nullptr == seq_buff && !padding_ids.empty()) {
    LOG(ERROR) << " Alloc sequence tensor memory fail.";
    return -1;
  }
  for (size_t i = 0; i < padding_ids.size(); ++i) {
    seq_buff[i * seq_len] = padding_ids[i];
  }

  bool res_status = false;
  if (std::fabs(p - 1.0) <= FLT_EPSILON && std::fabs(q - 1.0) <= FLT_EPSILON) {
    res_status = this->_SampleSeqWithoutBias(padding_ids, metapath, seq_buff);
  } else {
    res_status =
        this->_SampleSeqWithBias(padding_ids, metapath, p, q, seq_buff);
  }
  if (!res_status) {
    return -1;
  }
  return tensor_num;
}

bool DGraphImpl::_SampleNeighborWithoutWeightByMultiHop(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    const ArraySpec<uint32_t> &counts, size_t seq_len_per_vertex,
    VertexID *const neighbors_buff) const {
  std::vector<VertexID> tmp_ids;
  size_t hop_num = metapath.size();
  size_t child_num_per_vertex = 1;
  size_t collected_num_per_vertex = 1;
  ArraySpec<VertexID> cur_ids(ids.data, ids.cnt);
  bool res_status = false;
  for (size_t i = 0; i < hop_num; ++i) {
    Notifier notifier;
    size_t cur_cnt = counts.data[i];
    auto my_callback = [this, i, seq_len_per_vertex, child_num_per_vertex,
                        cur_cnt, neighbors_buff, collected_num_per_vertex,
                        &cur_ids, &notifier, &tmp_ids, &res_status, &metapath](
                           bool status, CollectNeighborResWithoutWeight &res) {
      res_status = status;
      if (status) {
        tmp_ids.clear();
        size_t vertex_idx = 0;
        size_t tmp_collected_num = collected_num_per_vertex;
        for (size_t j = 0; j < res.size(); ++j) {
          if (res[j].cnt != cur_cnt) {
            LOG(ERROR) << " The num of sampling neighbor is invalid."
                       << " id:" << cur_ids.data[j]
                       << " ,types:" << TypesToStr(metapath[i])
                       << " ,expected num:" << cur_cnt
                       << " ,real num:" << res[j].cnt;
            res_status = false;
            notifier.Notify();
            return;
          }
          if (vertex_idx != j / child_num_per_vertex) {
            tmp_collected_num = collected_num_per_vertex;
            vertex_idx = j / child_num_per_vertex;
          }
          size_t vertex_offset =
              seq_len_per_vertex * vertex_idx + tmp_collected_num;
          for (size_t k = 0; k < res[j].cnt; ++k) {
            size_t buff_idx = vertex_offset + k;
            neighbors_buff[buff_idx] = res[j].data[k];
            tmp_ids.push_back(res[j].data[k]);
          }
          tmp_collected_num += res[j].cnt;
        }
        cur_ids.cnt = tmp_ids.size();
        cur_ids.data = tmp_ids.data();
      }
      notifier.Notify();
    };
    graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithoutWeight,
                                 CollectNeighborResWithoutWeight>(
        galileo::common::SAMPLE_NEIGHBOR, cur_ids, metapath[i], counts.data[i],
        false, my_callback);
    notifier.WaitForNotification();
    if (!res_status) {
      return false;
    }
    child_num_per_vertex = child_num_per_vertex * cur_cnt;
    collected_num_per_vertex += child_num_per_vertex;
  }
  return res_status;
}

bool DGraphImpl::_SampleSeqWithBias(
    const std::vector<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, float p, float q,
    VertexID *const rw_sequences) const {
  std::vector<VertexID> parent_ids;
  std::vector<VertexID> child_ids;
  std::vector<VertexID> tmp_cids;
  std::vector<std::vector<galileo::common::IDWeight>> parents_neighbors;
  std::vector<std::vector<galileo::common::IDWeight>> childs_neighbors;
  child_ids.reserve(ids.size());
  size_t seq_num_per_vertex = metapath.size() + 1;
  for (size_t i = 0; i < ids.size(); ++i) {
    child_ids.push_back(ids[i]);
  }
  bool res_status = false;
  ArraySpec<VertexID> cur_ids;
  for (size_t i = 0; i < metapath.size(); ++i) {
    Notifier notifier;
    cur_ids.cnt = child_ids.size();
    cur_ids.data = child_ids.data();
    auto my_callback = [&childs_neighbors, &res_status, &notifier](
                           bool status, CollectNeighborResWithWeight &res) {
      res_status = status;
      if (status) {
        childs_neighbors.resize(res.size());
        for (size_t j = 0; j < res.size(); ++j) {
          auto &cur_neighbors = childs_neighbors.at(j);
          for (size_t k = 0; k < res[j].cnt; ++k) {
            cur_neighbors.push_back(res[j].data[k]);
          }
          auto cmp = [](galileo::common::IDWeight &pair1,
                        galileo::common::IDWeight &pair2) {
            return pair1.id_ < pair2.id_;
          };
          std::sort(cur_neighbors.begin(), cur_neighbors.end(), cmp);
        }
      }
      notifier.Notify();
    };
    graph_stub_->CollectNeighbor<galileo::common::NeighborReplyWithWeight,
                                 CollectNeighborResWithWeight>(
        galileo::common::GET_NEIGHBOR, cur_ids, metapath[i], 0, true,
        my_callback);
    notifier.WaitForNotification();
    if (!res_status) {
      return false;
    }
    tmp_cids.clear();
    for (size_t j = 0; j < childs_neighbors.size(); ++j) {
      if (0 == childs_neighbors[j].size()) {
        LOG(ERROR) << " The num of neighbors is zero."
                   << " id:" << cur_ids.data[j]
                   << " ,types:" << TypesToStr(metapath[i]);
        return false;
      }
      std::vector<float> weights;
      if (0 == parents_neighbors.size()) {
        this->_BuildWeightWithBias(childs_neighbors[j], &weights);
      } else {
        this->_BuildWeightWithBias(parent_ids[j], parents_neighbors[j],
                                   childs_neighbors[j], p, q, &weights);
      }
      galileo::common::SimpleSampler<galileo::common::IDWeight> sampler;
      sampler.Init(childs_neighbors[j], weights);
      VertexID tmp_neighbor = sampler.Sample().id_;
      size_t vertex_offset = j * seq_num_per_vertex + i + 1;
      rw_sequences[vertex_offset] = tmp_neighbor;
      tmp_cids.push_back(tmp_neighbor);
    }
    if (!res_status) {
      return false;
    }
    parent_ids = std::move(child_ids);
    parents_neighbors = std::move(childs_neighbors);
    child_ids = std::move(tmp_cids);
  }
  return res_status;
}

bool DGraphImpl::_SampleSeqWithoutBias(
    const std::vector<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    VertexID *const rw_sequences) const {
  std::vector<uint32_t> tmp_counts(metapath.size(), 1);
  ArraySpec<uint32_t> counts(tmp_counts.data(), tmp_counts.size());
  ArraySpec<VertexID> padding_ids(ids.data(), ids.size());
  size_t seq_len_per_vertex = metapath.size() + 1;
  return this->_SampleNeighborWithoutWeightByMultiHop(
      padding_ids, metapath, counts, seq_len_per_vertex, rw_sequences);
}

void DGraphImpl::_BuildRandomWalkSeqPair(const VertexID *seqs, int id_num,
                                         int seq_num_per_vertex,
                                         int context_size,
                                         VertexID *const pairs) const {
  int pair_idx = 0;
  for (int id_idx = 0; id_idx < id_num; ++id_idx) {
    int vertex_idx = seq_num_per_vertex * id_idx;
    for (int vertex_offset = 0; vertex_offset < seq_num_per_vertex;
         ++vertex_offset) {
      int r = 1;
      while ((vertex_offset + r < seq_num_per_vertex) && (r <= context_size)) {
        pairs[pair_idx++] = seqs[vertex_idx + vertex_offset];
        pairs[pair_idx++] = seqs[vertex_idx + vertex_offset + r];
        ++r;
      }
      int l = 1;
      while ((vertex_offset - l >= 0) && (l <= context_size)) {
        pairs[pair_idx++] = seqs[vertex_idx + vertex_offset];
        pairs[pair_idx++] = seqs[vertex_idx + vertex_offset - l];
        ++l;
      }
    }
  }
}

void DGraphImpl::_BuildWeightWithBias(
    const std::vector<galileo::common::IDWeight> &child_neighbors,
    std::vector<float> *weights) const {
  weights->clear();
  for (auto &neighbor : child_neighbors) {
    weights->push_back(neighbor.weight_);
  }
}

void DGraphImpl::_BuildWeightWithBias(
    VertexID parent_id,
    const std::vector<galileo::common::IDWeight> &parent_neighbors,
    const std::vector<galileo::common::IDWeight> &child_neighbors, float p,
    float q, std::vector<float> *weights) const {
  size_t c_idx = 0;
  size_t p_idx = 0;
  weights->clear();
  while (c_idx < child_neighbors.size() && p_idx < parent_neighbors.size()) {
    if (child_neighbors[c_idx].id_ < parent_neighbors[p_idx].id_) {
      if (child_neighbors[c_idx].id_ == parent_id) {
        weights->push_back(child_neighbors[c_idx].weight_ / p);
      } else {
        weights->push_back(child_neighbors[c_idx].weight_ / q);
      }
      ++c_idx;
    } else if (child_neighbors[c_idx].id_ == parent_neighbors[p_idx].id_) {
      weights->push_back(child_neighbors[c_idx].weight_);
      ++p_idx;
      ++c_idx;
    } else {
      ++p_idx;
    }
  }
  while (c_idx < child_neighbors.size()) {
    if (child_neighbors[c_idx].id_ != parent_id) {
      weights->push_back(child_neighbors[c_idx].weight_ / q);
    } else {
      weights->push_back(child_neighbors[c_idx].weight_ / p);
    }
    ++c_idx;
  }
}

size_t DGraphImpl::_GetMultiHopSeqNum(const ArraySpec<uint32_t> &counts) const {
  size_t total_num = 1;
  size_t cur_num = 1;
  for (size_t i = 0; i < counts.cnt; ++i) {
    cur_num = cur_num * counts.data[i];
    total_num += cur_num;
  }
  return total_num;
}

size_t DGraphImpl::_CalPairNum(size_t id_num, size_t repetition,
                               size_t walk_length, size_t context_size) const {
  return id_num * repetition * context_size *
         (2 * walk_length - context_size + 1);
}

std::string IDToStr(galileo::common::OperatorType op, const char *ids,
                    size_t idx) {
  std::stringstream stream;
  if (galileo::common::GET_VERTEX_FEATURE == op) {
    const ArraySpec<VertexID> *vertices = (const ArraySpec<VertexID> *)ids;
    stream << "[";
    stream << vertices->data[idx];
    stream << "]";
  } else {
    const EdgeArraySpec *edges = (const EdgeArraySpec *)ids;
    stream << "[";
    stream << edges->srcs.data[idx];
    stream << ",";
    stream << edges->dsts.data[idx];
    stream << ",";
    stream << (int)edges->types.data[idx];
    stream << "]";
  }
  return stream.str();
}

std::string TypesToStr(const ArraySpec<uint8_t> &types) {
  std::stringstream stream;
  stream << "[";
  for (size_t i = 0; i < types.cnt; ++i) {
    stream << (int)types.data[i];
    stream << ",";
  }
  stream << "]";
  return stream.str();
}

ClientType TransformDType2CType(galileo::proto::DataType dtype) {
  ClientType ctype = CT_INVALID_TYPE;
  switch (dtype) {
    case galileo::proto::DT_BOOL:
    case galileo::proto::DT_ARRAY_BOOL:
      ctype = CT_BOOL;
      break;
    case galileo::proto::DT_INT8:
    case galileo::proto::DT_ARRAY_INT8:
      ctype = CT_INT8;
      break;
    case galileo::proto::DT_UINT8:
    case galileo::proto::DT_ARRAY_UINT8:
      ctype = CT_UINT8;
      break;
    case galileo::proto::DT_INT16:
    case galileo::proto::DT_ARRAY_INT16:
      ctype = CT_INT16;
      break;
    case galileo::proto::DT_UINT16:
    case galileo::proto::DT_ARRAY_UINT16:
      ctype = CT_UINT16;
      break;
    case galileo::proto::DT_INT32:
    case galileo::proto::DT_ARRAY_INT32:
      ctype = CT_INT32;
      break;
    case galileo::proto::DT_UINT32:
    case galileo::proto::DT_ARRAY_UINT32:
      ctype = CT_UINT32;
      break;
    case galileo::proto::DT_INT64:
    case galileo::proto::DT_ARRAY_INT64:
      ctype = CT_INT64;
      break;
    case galileo::proto::DT_UINT64:
    case galileo::proto::DT_ARRAY_UINT64:
      ctype = CT_UINT64;
      break;
    case galileo::proto::DT_FLOAT:
    case galileo::proto::DT_ARRAY_FLOAT:
      ctype = CT_FLOAT;
      break;
    case galileo::proto::DT_DOUBLE:
    case galileo::proto::DT_ARRAY_DOUBLE:
      ctype = CT_DOUBLE;
      break;
    case galileo::proto::DT_STRING:
      ctype = CT_STRING;
      break;
    default:
      LOG(ERROR) << " Dtype is invalid.dtype:" << dtype;
      break;
  }
  return ctype;
}

size_t GetClientTypeCapacity(ClientType type) {
  size_t type_capacity = 0;
  switch (type) {
    case CT_BOOL:
      type_capacity = sizeof(bool);
      break;
    case CT_INT8:
      type_capacity = sizeof(int8_t);
      break;
    case CT_UINT8:
      type_capacity = sizeof(uint8_t);
      break;
    case CT_INT16:
      type_capacity = sizeof(int16_t);
      break;
    case CT_UINT16:
      type_capacity = sizeof(uint16_t);
      break;
    case CT_INT32:
      type_capacity = sizeof(int32_t);
      break;
    case CT_UINT32:
      type_capacity = sizeof(uint32_t);
      break;
    case CT_INT64:
      type_capacity = sizeof(int64_t);
      break;
    case CT_UINT64:
      type_capacity = sizeof(uint64_t);
      break;
    case CT_FLOAT:
      type_capacity = sizeof(float);
      break;
    case CT_DOUBLE:
      type_capacity = sizeof(double);
      break;
    default:
      LOG(ERROR) << " Ftype is invalid.type:" << type;
      return 0;
      break;
  }
  return type_capacity;
}

}  // namespace client
}  // namespace galileo
