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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

#include "../common/tensor_alloc.h"
#include "engine/client/dgraph_global.h"

using namespace galileo::client;

namespace tensorflow {
namespace glo {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

class CollectSeqByMultiHop : public AsyncOpKernel {
 public:
  explicit CollectSeqByMultiHop(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("counts", &counts_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &T_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
  bool OutputWeight();

 private:
  std::vector<int> counts_;
  DataTypeVector T_;
};

bool CollectSeqByMultiHop::OutputWeight() {
  for (auto ty : T_) {
    if (DT_FLOAT == ty) return true;
  }
  return false;
}

void CollectSeqByMultiHop::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument(" Global dgraph instance is nullptr.please "
                                "init global dgraph instance."),
        done);
    return;
  }
  auto tmp_ids = ctx->input(0);
  OpInputList tmp_metapath;
  ctx->input_list("metapath", &tmp_metapath);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(tmp_ids.shape()),
                    errors::InvalidArgument("ids must be a vector, shape:",
                                            tmp_ids.shape().DebugString()),
                    done);

  auto ids_value = tmp_ids.flat<int64>();
  ArraySpec<VertexID> ids((const int64_t*)ids_value.data(), ids_value.size());

  std::vector<ArraySpec<uint8_t>> metapath;
  for (int i = 0; i < tmp_metapath.size(); ++i) {
    auto metapath_value = tmp_metapath[i].flat<uint8>();
    metapath.emplace_back(metapath_value.data(), metapath_value.size());
  }
  ArraySpec<uint32_t> counts{reinterpret_cast<const uint32_t*>(counts_.data()),
                             counts_.size()};
  TFTypedTensorAlloc alloc(ctx, T_, 0);

  int res = gDGraph->CollectSeqByMultiHop(ids, metapath, counts,
                                          this->OutputWeight(), &alloc);
  int input_res = static_cast<int>(T_.size());
  OP_REQUIRES_ASYNC(
      ctx, res == input_res,
      errors::InvalidArgument(" Collect seq by multi hop is failed.input param "
                              "error or graph server error.res:",
                              res),
      done);
  done();
}

class CollectSeqByRWWithBias : public AsyncOpKernel {
 public:
  explicit CollectSeqByRWWithBias(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repetition", &repetition_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("p", &p_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("q", &q_));
    T_.push_back(DT_INT64);
  }
  virtual void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 protected:
  int repetition_;
  float p_;
  float q_;
  DataTypeVector T_;
};

void CollectSeqByRWWithBias::ComputeAsync(OpKernelContext* ctx,
                                          DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument(" Global dgraph instance is nullptr.please "
                                "init global dgraph instance."),
        done);
    return;
  }
  auto tmp_ids = ctx->input(0);
  OpInputList tmp_metapath;
  ctx->input_list("metapath", &tmp_metapath);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(tmp_ids.shape()),
                    errors::InvalidArgument(" Ids must be a vector, shape:",
                                            tmp_ids.shape().DebugString()),
                    done);

  auto ids_value = tmp_ids.flat<int64>();
  ArraySpec<VertexID> ids((const int64_t*)ids_value.data(), ids_value.size());

  std::vector<ArraySpec<uint8_t>> metapath;
  for (int i = 0; i < tmp_metapath.size(); ++i) {
    auto metapath_value = tmp_metapath[i].flat<uint8>();
    metapath.emplace_back(metapath_value.data(), metapath_value.size());
  }

  TFTypedTensorAlloc alloc(ctx, T_, 0);

  int res = gDGraph->CollectSeqByRWWithBias(
      ids, metapath, static_cast<uint32_t>(repetition_), p_, q_, &alloc);
  int expected_res = static_cast<int>(T_.size());
  OP_REQUIRES_ASYNC(ctx, res == expected_res,
                    errors::InvalidArgument(" Collect seq by random walk with "
                                            "bias is failed. input param error "
                                            "or graph server error.res:",
                                            res),
                    done);
  done();
}

class CollectPairByRWWithBias : public CollectSeqByRWWithBias {
 public:
  explicit CollectPairByRWWithBias(OpKernelConstruction* ctx)
      : CollectSeqByRWWithBias(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("context_size", &context_size_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int context_size_;
};

void CollectPairByRWWithBias::ComputeAsync(OpKernelContext* ctx,
                                           DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument("Global dgraph instance is nullptr.please init "
                                "global dgraph instance."),
        done);
    return;
  }
  auto tmp_ids = ctx->input(0);
  OpInputList tmp_metapath;
  ctx->input_list("metapath", &tmp_metapath);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(tmp_ids.shape()),
                    errors::InvalidArgument(" Ids must be a vector, shape:",
                                            tmp_ids.shape().DebugString()),
                    done);

  auto ids_value = tmp_ids.flat<int64>();
  ArraySpec<VertexID> ids((const int64_t*)ids_value.data(), ids_value.size());

  std::vector<ArraySpec<uint8_t>> metapath;
  for (int i = 0; i < tmp_metapath.size(); ++i) {
    auto metapath_value = tmp_metapath[i].flat<uint8>();
    metapath.emplace_back(metapath_value.data(), metapath_value.size());
  }

  TFTypedTensorAlloc alloc(ctx, T_, 0);

  int res = gDGraph->CollectPairByRWWithBias(
      ids, metapath, static_cast<uint32_t>(repetition_), p_, q_,
      static_cast<uint32_t>(context_size_), &alloc);
  int expected_res = static_cast<int>(T_.size());
  OP_REQUIRES_ASYNC(ctx, res == expected_res,
                    errors::InvalidArgument("Collect pair by random walk with "
                                            "bias is failed. input param error "
                                            "or graph server error.res:",
                                            res),
                    done);
  done();
}

}  // namespace glo
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(
    Name("CollectSeqByMultiHop").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectSeqByMultiHop);
REGISTER_KERNEL_BUILDER(
    Name("CollectSeqByRWWithBias").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectSeqByRWWithBias);
REGISTER_KERNEL_BUILDER(
    Name("CollectPairByRWWithBias").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectPairByRWWithBias);
