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

#include "../ops/ops.h"

#include "../common/tensor_alloc.h"
#include "engine/client/dgraph_global.h"

using namespace galileo::client;

namespace torch {
namespace glo {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

Tensors CollectSeqByMultiHop(Tensor ids, const Tensors& metapath,
                             const std::vector<int>& counts, bool has_weight) {
  if (nullptr == gDGraph) {
    LOG(ERROR) << " Global dgraph instance is nullptr.please init global "
                  "dgraph instance.";
    return {};
  }
  if (ids.dim() != 1 || metapath.size() == 0 ||
      metapath.size() != counts.size()) {
    LOG(ERROR) << " Collect seq by multi hop params error";
    return {};
  }

  auto ids_value = ids.data_ptr<int64_t>();
  ArraySpec<int64_t> ids_spec(ids_value, ids.numel());

  std::vector<ArraySpec<uint8_t>> paths_spec;
  for (size_t i = 0; i < metapath.size(); ++i) {
    auto path = metapath[i];
    paths_spec.emplace_back(path.data_ptr<uint8_t>(), path.numel());
  }

  ArraySpec<uint32_t> counts_spec(
      reinterpret_cast<const uint32_t*>(counts.data()), counts.size());

  Dtypes dtypes;
  dtypes.emplace_back(kLong);
  if (has_weight) {
    dtypes.emplace_back(kFloat);
  }

  Tensors tens;
  PTTypedTensorAlloc alloc(tens, dtypes);

  int res = gDGraph->CollectSeqByMultiHop(ids_spec, paths_spec, counts_spec,
                                          has_weight, &alloc);
  if (res != static_cast<int>(dtypes.size())) {
    LOG(ERROR) << " Collect seq by multi hop is failed.input param invalid or "
                  "graph server error."
               << " ,res:" << res;
    return {};
  }
  return tens;
}

Tensor CollectSeqByRWWithBias(Tensor ids, const Tensors& metapath,
                              int repetition, float p, float q) {
  if (nullptr == gDGraph) {
    LOG(ERROR) << " Global dgraph instance is nullptr.please init global "
                  "dgraph instance.";
    return {};
  }
  if (ids.dim() != 1 || metapath.size() == 0) {
    LOG(ERROR) << " Collect seq by rw with bias params error";
    return {};
  }

  auto ids_value = ids.data_ptr<int64_t>();
  ArraySpec<int64_t> ids_spec(ids_value, ids.numel());

  std::vector<ArraySpec<uint8_t>> paths_spec;
  for (size_t i = 0; i < metapath.size(); ++i) {
    auto path = metapath[i];
    paths_spec.emplace_back(path.data_ptr<uint8_t>(), path.numel());
  }

  Dtypes dtypes;
  dtypes.emplace_back(kLong);
  Tensors tens;
  PTTypedTensorAlloc alloc(tens, dtypes);

  int res = gDGraph->CollectSeqByRWWithBias(
      ids_spec, paths_spec, static_cast<uint32_t>(repetition), p, q, &alloc);
  if (res != (int)dtypes.size()) {
    LOG(ERROR) << " Collect seq by rw with bias is failed.input param invalid "
                  "or graph server error."
               << " ,res:" << res;
    return {};
  }
  return tens[0];
}

Tensor CollectPairByRWWithBias(Tensor ids, const Tensors& metapath,
                               int repetition, int context_size, float p,
                               float q) {
  if (nullptr == gDGraph) {
    LOG(ERROR) << " Global dgraph instance is nullptr.please init global "
                  "dgraph instance.";
    return {};
  }
  if (ids.dim() != 1 || metapath.size() == 0) {
    LOG(ERROR) << " Collect pair by rw with bias params error";
    return {};
  }

  auto ids_value = ids.data_ptr<int64_t>();
  ArraySpec<int64_t> ids_spec(ids_value, ids.numel());

  std::vector<ArraySpec<uint8_t>> paths_spec;
  for (size_t i = 0; i < metapath.size(); ++i) {
    auto path = metapath[i];
    paths_spec.emplace_back(path.data_ptr<uint8_t>(), path.numel());
  }

  Dtypes dtypes;
  dtypes.emplace_back(kLong);
  Tensors tens;
  PTTypedTensorAlloc alloc(tens, dtypes);

  int res = gDGraph->CollectPairByRWWithBias(
      ids_spec, paths_spec, static_cast<uint32_t>(repetition), p, q,
      static_cast<uint32_t>(context_size), &alloc);
  if (res != (int)dtypes.size()) {
    LOG(ERROR) << " Collect seq by rw with bias is failed.input param invalid "
                  "or graph server error."
               << " ,res:" << res;
    return {};
  }
  return tens[0];
}

}  // namespace glo
}  // namespace torch
