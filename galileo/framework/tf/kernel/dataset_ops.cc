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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

#include "../common/tensor_alloc.h"
#include "engine/client/dgraph.h"
#include "engine/client/dgraph_global.h"

using namespace galileo::client;

namespace tensorflow {
namespace glo {

template <typename T>
using ArraySpec = galileo::common::ArraySpec<T>;

constexpr const char* const kDatasetType = "EntityDatasetOp::Dataset";

class EntityDatasetOp : public DatasetOpKernel {
  int count_;
  std::string category_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;

 public:
  explicit EntityDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("category", &category_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    auto types = ctx->input(0);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(types.shape()),
                errors::InvalidArgument("types must be a vector, shape:",
                                        types.shape().DebugString()));

    *output = new Dataset(ctx, count_, category_, output_types_, output_shapes_,
                          types);
  }

 private:
  class Dataset : public DatasetBase {
   private:
    int count_;
    std::string category_;
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
    Tensor types_;

   public:
    explicit Dataset(OpKernelContext* ctx, int count, std::string& category,
                     const DataTypeVector& output_types,
                     const std::vector<PartialTensorShape>& output_shapes,
                     Tensor types)
        : DatasetBase(DatasetContext(ctx)),
          count_(count),
          category_(category),
          output_types_(output_types),
          output_shapes_(output_shapes),
          types_(types) {}

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return kDatasetType; }

    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* types_vertex = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(types_, &types_vertex));

      AttrValue count_attr;
      b->BuildAttrValue(count_, &count_attr);

      AttrValue category_attr;
      b->BuildAttrValue(category_, &category_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {types_vertex},
          {{"count", count_attr}, {"category", category_attr}}, output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<tensorflow::IteratorBase>(new Iterator(
          {this, tensorflow::strings::StrCat(prefix, kDatasetType)}));
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

     protected:
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        auto count = dataset()->count_;
        auto category = dataset()->category_;
        auto types = dataset()->types_;

        auto types_value = types.flat<uint8>();
        ArraySpec<uint8_t> spec(types_value.data(), types_value.size());

        TFDatasetTensorAlloc alloc(ctx, dataset()->output_types_, out_tensors);
        int res = 0;
        {
          mutex_lock l(mu_);
          if (nullptr == gDGraph) {
            return errors::InvalidArgument(
                " global dgraph instance is nullptr.please init global dgraph "
                "instance.");
          }
          res = gDGraph->CollectEntity(category, spec,
                                       static_cast<uint32_t>(count), &alloc);
        }
        int real_res = static_cast<int>(dataset()->output_types_.size());
        if (res != real_res) {
          return errors::InvalidArgument(
              " Entity dataset res is invalid. input param invalid or graph "
              "server error. res:",
              res);
        }
        *end_of_sequence = false;
        return Status::OK();
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return Status::OK();
      }

     private:
      tensorflow::mutex mu_;
    };
  };
};

}  // namespace glo
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("EntityDataset").Device(tensorflow::DEVICE_CPU),
                        tensorflow::glo::EntityDatasetOp);
