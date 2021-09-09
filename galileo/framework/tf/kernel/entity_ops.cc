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

class CollectEntity : public AsyncOpKernel {
 public:
  explicit CollectEntity(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("category", &category_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &T_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::string category_;
  DataTypeVector T_;
};

void CollectEntity::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  if (nullptr == gDGraph) {
    OP_REQUIRES_ASYNC(
        ctx, false,
        errors::InvalidArgument(" Global dgraph instance is nullptr.please "
                                "init global dgraph instance."),
        done);
    return;
  }
  auto types = ctx->input(0);
  auto count = ctx->input(1);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(types.shape()),
                    errors::InvalidArgument(" Types must be a vector, shape:",
                                            types.shape().DebugString()),
                    done);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(count.shape()),
              errors::InvalidArgument("count must be a scalar, saw shape: ",
                                      count.shape().DebugString()));

  auto types_value = types.flat<uint8>();
  auto count_value = (count.scalar<int32>())();
  ArraySpec<uint8_t> spec(types_value.data(), types_value.size());
  TFTypedTensorAlloc alloc(ctx, T_, 0);

  int res = gDGraph->CollectEntity(category_, spec,
                                   static_cast<uint32_t>(count_value), &alloc);
  int input_res = static_cast<int>(T_.size());
  OP_REQUIRES_ASYNC(
      ctx, res == input_res,
      errors::InvalidArgument(" Collect entity is failed.input param invalid "
                              "or graph server error.res:",
                              res),
      done);
  done();
}

}  // namespace glo
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("CollectEntity").Device(tensorflow::DEVICE_CPU),
                        tensorflow::glo::CollectEntity);
