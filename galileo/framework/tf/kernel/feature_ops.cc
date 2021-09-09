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

#include "../common/tensor_alloc.h"
#include "engine/client/dgraph_global.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace galileo::client;

namespace tensorflow {
namespace glo {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;
using EdgeArraySpec = galileo::common::EdgeArraySpec;

class CollectFeatureBase : public AsyncOpKernel {
 public:
  explicit CollectFeatureBase(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fnames", &fnames_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
  virtual int Collect(OpKernelContext* ctx, const std::string& category,
                      const char* ids,
                      const std::vector<ArraySpec<char>>& features,
                      const ArraySpec<uint32_t>& dims) = 0;

 protected:
  std::vector<std::string> fnames_;
  std::vector<int> dimensions_;
  int N_;
  int output_begin_idx_;
};

void CollectFeatureBase::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument(" Global dgraph instance is nullptr.please "
                                "init global dgraph instance."),
        done);
    return;
  }
  std::vector<ArraySpec<char>> features;
  for (size_t pos = 0; pos < fnames_.size(); ++pos) {
    features.emplace_back(fnames_[pos].data(), fnames_[pos].size());
  }

  ArraySpec<uint32_t> dims{reinterpret_cast<uint32_t*>(dimensions_.data()),
                           dimensions_.size()};

  int res = 0;
  size_t ids_count = 0;

  if (ctx->num_inputs() == 1) {
    auto ids = ctx->input(0);
    auto ids_value = ids.flat<int64>();
    ids_count = ids_value.size();
    ArraySpec<int64_t> spec((const int64_t*)ids_value.data(), ids_value.size());
    res = this->Collect(ctx, "vertex", (const char*)&spec, features, dims);
  } else {
    auto srcs = ctx->input(0);
    auto tars = ctx->input(1);
    auto types = ctx->input(2);

    auto srcs_value = srcs.flat<int64>();
    auto tars_value = tars.flat<int64>();
    auto types_value = types.flat<uint8>();

    ids_count = srcs_value.size();

    ArraySpec<int64_t> srcs_spec((const int64_t*)srcs_value.data(),
                                 srcs_value.size());
    ArraySpec<int64_t> tars_spec((const int64_t*)tars_value.data(),
                                 tars_value.size());
    ArraySpec<uint8> types_spec(types_value.data(), types_value.size());

    EdgeArraySpec spec(srcs_spec, tars_spec, types_spec);
    res = this->Collect(ctx, "edge", (const char*)&spec, features, dims);
  }
  if (0 == ids_count) {
    // alloc empty tensors
    for (int i = output_begin_idx_; i < N_ + output_begin_idx_; ++i) {
      Tensor* tensor = nullptr;
      auto dim = dimensions_[i - output_begin_idx_];
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(i, {0L, dim}, &tensor),
                           done);
    }
  }

  int input_res = N_;
  OP_REQUIRES_ASYNC(
      ctx, res == input_res,
      errors::InvalidArgument(" Collect feature is failed.input param invalid "
                              "or graph server error.res:",
                              res),
      done);
  done();
}

class CollectFeature : public CollectFeatureBase {
 public:
  explicit CollectFeature(OpKernelConstruction* ctx) : CollectFeatureBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("TO", &TO_));
    N_ = TO_.size();
    output_begin_idx_ = 0;
  }

  int Collect(OpKernelContext* ctx, const std::string& category,
              const char* ids, const std::vector<ArraySpec<char>>& features,
              const ArraySpec<uint32_t>& dims) override {
    TFTypedTensorAlloc alloc(ctx, TO_, 0);
    return gDGraph->CollectFeature(category, ids, features, dims, &alloc);
  }

 private:
  DataTypeVector TO_;
};

class CollectPodFeature : public CollectFeatureBase {
 public:
  explicit CollectPodFeature(OpKernelConstruction* ctx)
      : CollectFeatureBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    output_begin_idx_ = 1;
  }

  int Collect(OpKernelContext* ctx, const std::string& category,
              const char* ids, const std::vector<ArraySpec<char>>& features,
              const ArraySpec<uint32_t>& dims) override {
    TFBitTensorAlloc alloc(ctx, 1, N_, 0);
    return gDGraph->CollectPodFeature(category, ids, features, dims, &alloc);
  }
};

class CollectStringFeature : public CollectFeatureBase {
 public:
  explicit CollectStringFeature(OpKernelConstruction* ctx)
      : CollectFeatureBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    output_begin_idx_ = 0;
  }

  int Collect(OpKernelContext* ctx, const std::string& category,
              const char* ids, const std::vector<ArraySpec<char>>& features,
              const ArraySpec<uint32_t>& dims) override {
    DataTypeVector vec(N_, DT_STRING);
    TFTypedTensorAlloc alloc(ctx, vec, 0);
    return gDGraph->CollectFeature(category, ids, features, dims, &alloc);
  }
};

}  // namespace glo
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("CollectFeature").Device(tensorflow::DEVICE_CPU),
                        tensorflow::glo::CollectFeature);
REGISTER_KERNEL_BUILDER(
    Name("CollectPodFeature").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectPodFeature);
REGISTER_KERNEL_BUILDER(
    Name("CollectStringFeature").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectStringFeature);
