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

class CollectNeighborBase : public AsyncOpKernel {
 public:
  explicit CollectNeighborBase(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &T_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
  bool OutputWeight();

  virtual int Collect(OpKernelContext* ctx, const ArraySpec<int64_t>& ids,
                      const ArraySpec<uint8>& types, uint32_t count) = 0;

 protected:
  int count_;
  DataTypeVector T_;
};

bool CollectNeighborBase::OutputWeight() {
  for (auto ty : T_) {
    if (DT_FLOAT == ty) return true;
  }
  return false;
}

void CollectNeighborBase::ComputeAsync(OpKernelContext* ctx,
                                       DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument(" Global dgraph instance is nullptr.please "
                                "init global dgraph instance."),
        done);
    return;
  }
  auto ids = ctx->input(0);
  auto types = ctx->input(1);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(ids.shape()),
                    errors::InvalidArgument(" Ids must be a vector, shape:",
                                            ids.shape().DebugString()),
                    done);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(types.shape()),
                    errors::InvalidArgument(" Types must be a vector, shape:",
                                            types.shape().DebugString()),
                    done);

  auto ids_value = ids.flat<int64>();
  auto types_value = types.flat<uint8>();

  ArraySpec<int64_t> ids_spec((const int64_t*)ids_value.data(),
                              ids_value.size());
  ArraySpec<uint8> types_spec(types_value.data(), types_value.size());

  int res =
      this->Collect(ctx, ids_spec, types_spec, static_cast<uint32_t>(count_));
  int input_res = static_cast<int>(T_.size());
  OP_REQUIRES_ASYNC(
      ctx, res == input_res,
      errors::InvalidArgument(" Collect neighbor is failed.input param invalid "
                              "or graph server error.res:",
                              res),
      done);
  done();
}

class CollectNeighbor : public CollectNeighborBase {
 public:
  explicit CollectNeighbor(OpKernelConstruction* ctx)
      : CollectNeighborBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("category", &category_));
  }

  int Collect(OpKernelContext* ctx, const ArraySpec<int64_t>& ids,
              const ArraySpec<uint8>& types, uint32_t count) override {
    TFTypedTensorAlloc alloc(ctx, T_, 0);
    return gDGraph->CollectNeighbor(category_, ids, types, count,
                                    this->OutputWeight(), &alloc);
  }

 private:
  std::string category_;
};

class CollectStateNeighbor : public CollectNeighborBase {
 public:
  explicit CollectStateNeighbor(OpKernelConstruction* ctx)
      : CollectNeighborBase(ctx) {}

  int Collect(OpKernelContext* ctx, const ArraySpec<int64_t>& ids,
              const ArraySpec<uint8>& types, uint32_t count) override {
    TFTypedTensorAlloc alloc(ctx, T_, 0);
    return gDGraph->CollectNeighbor("sample", ids, types, count,
                                    this->OutputWeight(), &alloc);
  }
};

}  // namespace glo
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("CollectNeighbor").Device(tensorflow::DEVICE_CPU),
                        tensorflow::glo::CollectNeighbor);
REGISTER_KERNEL_BUILDER(
    Name("CollectStateNeighbor").Device(tensorflow::DEVICE_CPU),
    tensorflow::glo::CollectStateNeighbor);
