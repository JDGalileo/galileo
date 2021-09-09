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

#include "client/dgraph.h"
#include "client/dgraph_impl.h"

namespace galileo {
namespace client {

DGraph::DGraph() : graph_impl_(new DGraphImpl()) {}

DGraph::~DGraph() {}

bool DGraph::Initialize(const DGraphConfig &config) {
  return graph_impl_->Initialize(config);
}

int DGraph::CollectEntity(const std::string &category,
                          const ArraySpec<uint8_t> &types, uint32_t count,
                          ITensorAlloc *alloc) const {
  return graph_impl_->CollectEntity(category, types, count, alloc);
}

int DGraph::CollectNeighbor(const std::string &category,
                            const ArraySpec<VertexID> &ids,
                            const ArraySpec<uint8_t> &edge_types,
                            uint32_t count, bool need_weight,
                            ITensorAlloc *alloc) const {
  return graph_impl_->CollectNeighbor(category, ids, edge_types, count,
                                      need_weight, alloc);
}

int DGraph::CollectFeature(const std::string &category, const char *ids,
                           const std::vector<ArraySpec<char>> &features_name,
                           const ArraySpec<uint32_t> &max_dims,
                           ITensorAlloc *alloc) const {
  return graph_impl_->CollectFeature(category, ids, features_name, max_dims,
                                     alloc);
}

int DGraph::CollectPodFeature(const std::string &category, const char *ids,
                              const std::vector<ArraySpec<char>> &features_name,
                              const ArraySpec<uint32_t> &max_dims,
                              ITensorAlloc *alloc) const {
  return graph_impl_->CollectPodFeature(category, ids, features_name, max_dims,
                                        alloc);
}

int DGraph::CollectSeqByMultiHop(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath,
    const ArraySpec<uint32_t> &counts, bool need_weight,
    ITensorAlloc *alloc) const {
  return graph_impl_->CollectSeqByMultiHop(ids, metapath, counts, need_weight,
                                           alloc);
}

int DGraph::CollectSeqByRWWithBias(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
    float p, float q, ITensorAlloc *alloc) const {
  return graph_impl_->CollectSeqByRWWithBias(ids, metapath, repetition, p, q,
                                             alloc);
}

int DGraph::CollectPairByRWWithBias(
    const ArraySpec<VertexID> &ids,
    const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
    float p, float q, uint32_t context_size, ITensorAlloc *alloc) const {
  return graph_impl_->CollectPairByRWWithBias(ids, metapath, repetition, p, q,
                                              context_size, alloc);
}

bool DGraph::CollectGraphMeta(GraphMeta *meta_info) const {
  return graph_impl_->CollectGraphMeta(meta_info);
}

}  // namespace client
}  // namespace galileo
