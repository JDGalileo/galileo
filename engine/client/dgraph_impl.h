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
#include <vector>

#include "client/dgraph_stub.h"
#include "client/dgraph_type.h"
#include "common/types.h"
#include "proto/types.pb.h"

namespace galileo {
namespace client {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

class DGraphImpl {
 public:
  DGraphImpl();
  ~DGraphImpl();

  bool Initialize(const DGraphConfig &config);

  int CollectEntity(const std::string &op, const ArraySpec<uint8_t> &types,
                    uint32_t count, ITensorAlloc *alloc) const;

  int CollectNeighbor(const std::string &op, const ArraySpec<VertexID> &ids,
                      const ArraySpec<uint8_t> &types, uint32_t count,
                      bool need_weight, ITensorAlloc *alloc) const;

  int CollectFeature(const std::string &op, const char *ids,
                     const std::vector<ArraySpec<char>> &features_name,
                     const ArraySpec<uint32_t> &max_dims,
                     ITensorAlloc *alloc) const;

  int CollectPodFeature(const std::string &op, const char *ids,
                        const std::vector<ArraySpec<char>> &features_name,
                        const ArraySpec<uint32_t> &max_dims,
                        ITensorAlloc *alloc) const;

  int CollectSeqByMultiHop(const ArraySpec<VertexID> &ids,
                           const std::vector<ArraySpec<uint8_t>> &metapath,
                           const ArraySpec<uint32_t> &counts, bool need_weight,
                           ITensorAlloc *alloc) const;

  int CollectSeqByRWWithBias(const ArraySpec<VertexID> &ids,
                             const std::vector<ArraySpec<uint8_t>> &metapath,
                             uint32_t repetition, float p, float q,
                             ITensorAlloc *alloc) const;

  int CollectPairByRWWithBias(const ArraySpec<VertexID> &ids,
                              const std::vector<ArraySpec<uint8_t>> &metapath,
                              uint32_t repetition, float p, float q,
                              uint32_t context_size, ITensorAlloc *alloc) const;

  bool CollectGraphMeta(GraphMeta *meta_info) const;

 private:
  int _GetLimitedNeighborWithWeight(galileo::common::OperatorType op,
                                    const ArraySpec<VertexID> &ids,
                                    const ArraySpec<uint8_t> &edge_types,
                                    uint32_t count, ITensorAlloc *alloc) const;

  int _GetLimitedNeighborWithoutWeight(galileo::common::OperatorType op,
                                       const ArraySpec<VertexID> &ids,
                                       const ArraySpec<uint8_t> &edge_types,
                                       uint32_t count,
                                       ITensorAlloc *alloc) const;

  int _GetFullNeighborWithWeight(const ArraySpec<VertexID> &ids,
                                 const ArraySpec<uint8_t> &edge_types,
                                 uint32_t count, ITensorAlloc *alloc) const;

  int _GetFullNeighborWithoutWeight(const ArraySpec<VertexID> &ids,
                                    const ArraySpec<uint8_t> &edge_types,
                                    uint32_t count, ITensorAlloc *alloc) const;

  int _GetFeature(galileo::common::OperatorType op, const char *ids,
                  const std::vector<ArraySpec<char>> &features,
                  const ArraySpec<uint32_t> &max_dims,
                  ITensorAlloc *alloc) const;

  int _GetPodFeature(galileo::common::OperatorType op, const char *ids,
                     const std::vector<ArraySpec<char>> &features,
                     const ArraySpec<uint32_t> &max_dims,
                     ITensorAlloc *alloc) const;

  int _SampleSeqWithWeightByMultiHop(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath,
      const ArraySpec<uint32_t> &counts, ITensorAlloc *alloc) const;

  int _SampleSeqWithoutWeightByMultiHop(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath,
      const ArraySpec<uint32_t> &counts, ITensorAlloc *alloc) const;

  bool _SampleNeighborWithoutWeightByMultiHop(
      const ArraySpec<VertexID> &ids,
      const std::vector<ArraySpec<uint8_t>> &metapath,
      const ArraySpec<uint32_t> &counts, size_t seq_len_per_vertex,
      VertexID *const neighbors_buff) const;

  int _SampleSeqByRWWithBias(const ArraySpec<VertexID> &ids,
                             const std::vector<ArraySpec<uint8_t>> &metapath,
                             uint32_t repetition, float p, float q,
                             ITensorAlloc *alloc) const;

  bool _SampleSeqWithBias(const std::vector<VertexID> &ids,
                          const std::vector<ArraySpec<uint8_t>> &metapath,
                          float p, float q, VertexID *const rw_sequences) const;

  bool _SampleSeqWithoutBias(const std::vector<VertexID> &ids,
                             const std::vector<ArraySpec<uint8_t>> &metapath,
                             VertexID *const rw_sequences) const;

  void _BuildRandomWalkSeqPair(const VertexID *seqs, int id_num,
                               int seq_num_per_vertex, int context_size,
                               VertexID *const pairs) const;

  void _BuildWeightWithBias(
      VertexID parent_id,
      const std::vector<galileo::common::IDWeight> &parent_neighbors,
      const std::vector<galileo::common::IDWeight> &child_neighbors, float p,
      float q, std::vector<float> *weights) const;

  void _BuildWeightWithBias(
      const std::vector<galileo::common::IDWeight> &child_neighbors,
      std::vector<float> *weights) const;

  size_t _GetMultiHopSeqNum(const ArraySpec<uint32_t> &counts) const;

  size_t _CalPairNum(size_t id_num, size_t repetition, size_t walk_length,
                     size_t context_size) const;

  size_t _GetIDCnt(galileo::common::OperatorType op, const char *ids) const;

 private:
  std::unique_ptr<DGraphStub> graph_stub_;
};

ClientType TransformDType2CType(galileo::proto::DataType dtype);
size_t GetClientTypeCapacity(ClientType type);
std::string IDToStr(galileo::common::OperatorType op, const char *ids,
                    size_t idx);
std::string TypesToStr(const ArraySpec<uint8_t> &types);

}  // namespace client
}  // namespace galileo
