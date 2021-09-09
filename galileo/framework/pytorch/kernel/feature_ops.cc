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
using EdgeArraySpec = galileo::common::EdgeArraySpec;

Tensors CollectPodFeature(Tensors ids, Fnames& fnames, Dims& dimensions) {
  if (nullptr == gDGraph) {
    LOG(ERROR) << " Global dgraph instance is nullptr.please init global "
                  "dgraph instance.";
    return {};
  }
  if ((ids.size() != 1 && ids.size() != 3) ||
      (ids.size() == 1 && ids[0].scalar_type() != kLong) ||
      (ids.size() == 3 &&
       (ids[0].scalar_type() != kLong || ids[1].scalar_type() != kLong ||
        ids[2].scalar_type() != kByte)) ||
      fnames.size() == 0 || fnames.size() != dimensions.size()) {
    LOG(ERROR) << " Collect pod feature input params error";
    return {};
  }

  std::vector<ArraySpec<char>> features;
  for (size_t pos = 0; pos < fnames.size(); ++pos) {
    features.emplace_back(fnames[pos].data(), fnames[pos].size());
  }

  ArraySpec<uint32_t> dims(reinterpret_cast<uint32_t*>(dimensions.data()),
                           dimensions.size());
  Tensors tens;
  PTAnyTensorAlloc alloc(tens);

  int res = 0;
  if (ids.size() == 1) {
    std::string category("vertex");
    auto ids_value = ids[0].data_ptr<int64_t>();
    ArraySpec<int64_t> spec(ids_value, ids[0].numel());
    res = gDGraph->CollectFeature(category, (const char*)&spec, features, dims,
                                  &alloc);
  } else {
    std::string category("edge");
    auto srcs_value = ids[0].data_ptr<int64_t>();
    auto tars_value = ids[1].data_ptr<int64_t>();
    auto types_value = ids[2].data_ptr<uint8_t>();

    ArraySpec<int64_t> srcs_spec(srcs_value, ids[0].numel());
    ArraySpec<int64_t> tars_spec(tars_value, ids[1].numel());
    ArraySpec<uint8_t> types_spec(types_value, ids[2].numel());

    EdgeArraySpec spec(srcs_spec, tars_spec, types_spec);
    res = gDGraph->CollectFeature(category, (const char*)&spec, features, dims,
                                  &alloc);
  }
  if (res != static_cast<int>(tens.size())) {
    LOG(ERROR) << " Collect pod feature is failed.input param invalid or graph "
                  "server error"
               << " ,res:" << res;
    return {};
  }
  return tens;
}

Tensors CollectStringFeature(Tensors ids, Fnames& fnames, Dims& dimensions) {
  Tensors tens;
  return tens;
}

}  // namespace glo
}  // namespace torch
