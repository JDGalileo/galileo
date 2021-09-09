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

#ifndef __Tensor_Alloc_H__
#define __Tensor_Alloc_H__

#include "engine/client/dgraph_type.h"
#include "types_convert.h"

using namespace galileo::client;

namespace torch {
namespace glo {

class PTTensorAllocBase : public ITensorAlloc {
 public:
  int GetTensorType(ClientType type) override {
    return static_cast<int>(EngineType2PT(type));
  }

  bool FillStringTensor(char* buffer, size_t idx,
                        const ArraySpec<char>& str) override {
    return true;
  }

  bool FillStringTensor(char* buffer, size_t idx,
                        const std::string& str) override {
    return true;
  }
};

class PTTypedTensorAlloc : public PTTensorAllocBase {
 public:
  PTTypedTensorAlloc(Tensors& tensors, const Dtypes& types)
      : tensors_(tensors), types_(types) {}

  virtual ~PTTypedTensorAlloc() {}

  char* AllocListTensor(ClientType type,
                        const std::initializer_list<long long>& dims) override {
    if (types_.size() == tensors_.size()) return nullptr;

    Dtype ty = EngineType2PT(type);
    if (Dtype::Undefined == ty || types_[tensors_.size()] != ty) return nullptr;

    Tensor tensor = torch::empty({(int64_t*)dims.begin(), dims.size()}, ty);
    tensors_.emplace_back(std::move(tensor));

    return (char*)tensors_.back().data_ptr();
  }

 private:
  Tensors& tensors_;
  const Dtypes& types_;
};

class PTAnyTensorAlloc : public PTTensorAllocBase {
 public:
  PTAnyTensorAlloc(Tensors& tensors) : tensors_(tensors) {}

  virtual ~PTAnyTensorAlloc() {}

  char* AllocListTensor(ClientType type,
                        const std::initializer_list<long long>& dims) override {
    Dtype ty = EngineType2PT(type);
    if (Dtype::Undefined == ty) return nullptr;

    Tensor tensor = torch::empty({(int64_t*)dims.begin(), dims.size()}, ty);
    tensors_.emplace_back(std::move(tensor));

    return (char*)tensors_.back().data_ptr();
  }

 private:
  Tensors& tensors_;
};

}  // namespace glo
}  // namespace torch

#endif  // __Tensor_Alloc_H__
