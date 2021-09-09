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

#include <memory>
#include <string>

#include "client/dgraph_type.h"
#include "common/macro.h"
#include "common/types.h"

namespace galileo {
namespace client {

class DGraphImpl;
class DGraphConfig;

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

class DGraph {
 public:
  DGraph();
  ~DGraph();

  bool Initialize(const DGraphConfig &config);

  CLIENT_EXTERNAL int CollectEntity(const std::string &op,
                                    const ArraySpec<uint8_t> &types,
                                    uint32_t count, ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectNeighbor(const std::string &op,
                                      const ArraySpec<VertexID> &ids,
                                      const ArraySpec<uint8_t> &types,
                                      uint32_t count, bool need_weight,
                                      ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectFeature(
      const std::string &op, const char *ids,
      const std::vector<ArraySpec<char>> &features_name,
      const ArraySpec<uint32_t> &max_dims, ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectPodFeature(
      const std::string &op, const char *ids,
      const std::vector<ArraySpec<char>> &features_name,
      const ArraySpec<uint32_t> &max_dims, ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectSeqByMultiHop(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath,
      const ArraySpec<uint32_t> &counts, bool need_weight,
      ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectSeqByRWWithBias(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
      float p, float q, ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL int CollectPairByRWWithBias(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath, uint32_t repetition,
      float p, float q, uint32_t context_size, ITensorAlloc *alloc) const;

  CLIENT_EXTERNAL bool CollectGraphMeta(GraphMeta *meta_info) const;

 private:
  std::unique_ptr<DGraphImpl> graph_impl_;
};

}  // namespace client
}  // namespace galileo
