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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"

REGISTER_OP("CollectStateNeighbor")
    .Input("ids: int64")
    .Input("types: uint8")
    .Attr("count: int >= 1")
    .Attr("T: list({int64, float})")
    .Output("outs: T")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &types));

      int count = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("count", &count));
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((T.size() != 1 && T.size() != 2) ||
          (T.size() == 1 && T[0] != tensorflow::DT_INT64) ||
          (T.size() == 2 &&
           (T[0] != tensorflow::DT_INT64 || T[1] != tensorflow::DT_FLOAT))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid input parm T");
      }

      auto shape = c->MakeShape({c->Dim(ids, 0), c->MakeDim(count)});
      for (size_t idx = 0; idx < T.size(); ++idx) {
        c->set_output(idx, shape);
      }

      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectStateNeighbor

collect neighbor op, for sample

ids: input, vertex id list
types: input, edge type list
count: attr, neighbor count
T: attr, result type list [DT_INT64, DT_FLOAT], DT_FLOAT is optional
outs: output,  output list

)doc");

REGISTER_OP("CollectNeighbor")
    .Input("ids: int64")
    .Input("types: uint8")
    .Attr("count: int")
    .Attr("category: {'topk', 'full'}")
    .Attr("T: list({int64, float, int32})")
    .Output("outs: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &types));

      int count = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("count", &count));
      std::string category;
      TF_RETURN_IF_ERROR(c->GetAttr("category", &category));
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if (category == "topk") {
        if ((T.size() != 1 && T.size() != 2) ||
            (T.size() == 1 && T[0] != tensorflow::DT_INT64) ||
            (T.size() == 2 && T[0] != tensorflow::DT_INT64 &&
             T[1] != tensorflow::DT_FLOAT)) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    " Invalid attr T");
        }
      } else if (category == "full") {
        if ((T.size() != 2 && T.size() != 3) ||
            (T.size() == 2 && T[0] != tensorflow::DT_INT64 &&
             T[1] != tensorflow::DT_INT32) ||
            (T.size() == 3 && T[0] != tensorflow::DT_INT64 &&
             T[1] != tensorflow::DT_FLOAT && T[2] != tensorflow::DT_INT32)) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    " Invalid attr T");
        }
      } else {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr category");
      }
      auto data_shape = c->UnknownShapeOfRank(1);
      auto idx_shape = c->MakeShape({c->Dim(ids, 0), 2});
      for (size_t idx = 0; idx < T.size(); ++idx) {
        if (T[idx] == tensorflow::DT_FLOAT || T[idx] == tensorflow::DT_INT64)
          c->set_output(idx, data_shape);
        else
          c->set_output(idx, idx_shape);
      }

      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectNeighbor

collect neighbor op, for topk full and so on

ids: input, vertex id list
types: input, edge type list
count: attr, neighbor count.if category is full,count is 0;if category is topk,count is greater than 0
category: attr, neighbor enum type
T: attr, result type list [DT_INT64, DT_FLOAT, DT_INT32]
outs: output,  output list 

)doc");
