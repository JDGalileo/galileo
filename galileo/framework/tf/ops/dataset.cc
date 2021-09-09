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

REGISTER_OP("EntityDataset")
    .Input("types: uint8")
    .Attr("count: int >= 1")
    .Attr("category: {'vertex', 'edge'}")
    .Attr("output_types: list({int64, uint8}) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &types));

      int count = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("count", &count));
      std::string category;
      TF_RETURN_IF_ERROR(c->GetAttr("category", &category));
      tensorflow::DataTypeVector output_types;
      TF_RETURN_IF_ERROR(c->GetAttr("output_types", &output_types));
      if ((category == "vertex" && (output_types.size() != 1 ||
                                    output_types[0] != tensorflow::DT_INT64)) ||
          (category == "edge" && (output_types.size() != 3 ||
                                  output_types[0] != tensorflow::DT_INT64 ||
                                  output_types[1] != tensorflow::DT_INT64 ||
                                  output_types[2] != tensorflow::DT_UINT8))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr output_types");
      }

      c->set_output(0, c->Scalar());
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
EntityDataset

entity dataset op, include vertex edge and so on

types: input, type list
count: attr, count for types
category: attr, entity enum type
output_types: attr, output type list [DT_INT64] or [DT_INT64, DT_INT64, DT_UINT8]
output_shapes: attr, output shapes list
handle: entity dataset handle

)doc");
