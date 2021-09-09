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

#include <iostream>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("CollectEntity")
    .Input("types: uint8")
    .Input("count: int32")
    .Attr("category: {'vertex', 'edge'}")
    .Attr("T: list({int64, uint8})")
    .Output("outs: T")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle types;
      tensorflow::shape_inference::ShapeHandle unused;
      tensorflow::shape_inference::DimensionHandle count;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &types));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &count));

      std::string category;
      TF_RETURN_IF_ERROR(c->GetAttr("category", &category));
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((category == "vertex" &&
           (T.size() != 1 || T[0] != tensorflow::DT_INT64)) ||
          (category == "edge" &&
           (T.size() != 3 || T[0] != tensorflow::DT_INT64 ||
            T[1] != tensorflow::DT_INT64 || T[2] != tensorflow::DT_UINT8))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr T");
      }
      auto shp = c->MakeShape({c->Dim(types, 0), count});
      if (category == "vertex") {
        c->set_output(0, shp);
      } else if (category == "edge") {
        c->set_output(0, shp);
        c->set_output(1, shp);
        c->set_output(2, shp);
      }

      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectEntity

collect graph entity op, include vertex edge and so on

types: input, type list
count: input, count for types
category: attr, entity enum type
T: attr, result type list [DT_INT64] or [DT_INT64, DT_INT64, DT_UINT8]
outs: output,  output list
)doc");
