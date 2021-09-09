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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "types_convert.h"

#include "engine/client/dgraph_type.h"
using namespace galileo::client;

namespace tensorflow {
namespace glo {

class TFTensorAllocBase : public ITensorAlloc {
 public:
  int GetTensorType(ClientType type) override { return EngineType2TF(type); }

  bool FillStringTensor(char* buffer, size_t idx,
                        const ArraySpec<char>& str) override {
    tstring* arr = (tstring*)buffer;
    arr[idx] = std::move(tstring(str.data, str.cnt));
    return true;
  }

  bool FillStringTensor(char* buffer, size_t idx,
                        const std::string& str) override {
    tstring* arr = (tstring*)buffer;
    arr[idx] = std::move(tstring(str));
    return true;
  }
};

class TFTypedTensorAlloc : public TFTensorAllocBase {
 public:
  TFTypedTensorAlloc(OpKernelContext* ctx, const DataTypeVector& types,
                     size_t list_begin)
      : ctx_(ctx), types_(types), list_begin_(list_begin), seq_num_(0) {}

  virtual ~TFTypedTensorAlloc() {
    ctx_ = nullptr;
    list_begin_ = seq_num_ = 0;
  }

  char* AllocListTensor(ClientType type,
                        const std::initializer_list<long long>& dims) override {
    if (types_.size() == static_cast<size_t>(seq_num_)) return nullptr;

    DataType ty = EngineType2TF(type);
    if (DT_INVALID == ty || types_[seq_num_] != ty) return nullptr;

    Tensor* tensor = nullptr;
    Status res = ctx_->allocate_output(list_begin_ + seq_num_++, dims, &tensor);

    if (!TF_PREDICT_TRUE(res.ok())) {
      ctx_->CtxFailureWithWarning(__FILE__, __LINE__, res);
      return nullptr;
    }

    return static_cast<char*>(tensor->data());
  }

 private:
  OpKernelContext* ctx_;
  const DataTypeVector& types_;

  int list_begin_;
  int seq_num_;
};

class TFBitTensorAlloc : public TFTensorAllocBase {
 public:
  TFBitTensorAlloc(OpKernelContext* ctx, int list_begin, int list_num,
                   int type_pos = -1)
      : ctx_(ctx),
        list_begin_(list_begin),
        list_num_(list_num),
        type_pos_(type_pos),
        seq_num_(0) {}

  virtual ~TFBitTensorAlloc() {
    ctx_ = nullptr;
    list_begin_ = list_num_ = type_pos_ = seq_num_ = 0;
  }

  char* AllocListTensor(ClientType type,
                        const std::initializer_list<long long>& dims) override {
    if (list_num_ == seq_num_) return nullptr;

    DataType ty = EngineType2TF(type);
    if (DT_INVALID == ty) return nullptr;

    size_t len = DataTypeSize(ty);
    if (0 == len) return nullptr;

    TensorShape shape;
    for (auto it = dims.begin(); it != dims.end(); ++it) {
      shape.AddDim(*it);
    }
    shape.AddDim(len);

    Tensor* tensor = nullptr;
    Status sta =
        ctx_->allocate_output(list_begin_ + seq_num_++, shape, &tensor);

    if (!TF_PREDICT_TRUE(sta.ok())) {
      ctx_->CtxFailureWithWarning(__FILE__, __LINE__, sta);
      return nullptr;
    }

    return static_cast<char*>(tensor->data());
  }

  char* AllocTypesTensor(long long count) override {
    if (-1 == type_pos_) return nullptr;

    Tensor* tensor = nullptr;
    Status sta = ctx_->allocate_output(type_pos_, {count}, &tensor);

    if (!TF_PREDICT_TRUE(sta.ok())) {
      ctx_->CtxFailureWithWarning(__FILE__, __LINE__, sta);
      return nullptr;
    }

    return static_cast<char*>(tensor->data());
  }

 private:
  OpKernelContext* ctx_;

  int list_begin_;
  int list_num_;

  int type_pos_;
  int seq_num_;
};

class TFDatasetTensorAlloc : public TFTensorAllocBase {
 public:
  TFDatasetTensorAlloc(IteratorContext* ctx, const DataTypeVector& types,
                       std::vector<Tensor>* out_tensors)
      : ctx_(ctx), types_(types), out_tensors_(out_tensors) {}

  virtual ~TFDatasetTensorAlloc() {
    ctx_ = nullptr;
    out_tensors_ = nullptr;
  }

  char* AllocListTensor(ClientType type,
                        const std::initializer_list<long long>& dims) override {
    if (types_.size() == out_tensors_->size()) return nullptr;

    DataType ty = EngineType2TF(type);
    if (DT_INVALID == ty || types_[out_tensors_->size()] != ty) return nullptr;

    out_tensors_->emplace_back(ctx_->allocator({}), ty, dims);
    Tensor* tensor = &out_tensors_->back();

    return static_cast<char*>(tensor->data());
  }

 private:
  IteratorContext* ctx_;
  const DataTypeVector& types_;

  std::vector<Tensor>* out_tensors_;
};

}  // namespace glo
}  // namespace tensorflow

#endif  // __Tensor_Alloc_H__
