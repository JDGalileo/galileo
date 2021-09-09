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
#include "tensorflow/core/framework/types.pb.h"

REGISTER_OP("CollectSeqByMultiHop")
    .Input("ids: int64")
    .Input("metapath: hop_num * uint8")
    .Attr("counts: list(int)")
    .Attr("hop_num: int >= 1")
    .Attr("T: list({int64, float})")
    .Output("outs: T")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &types));
      std::vector<int> counts;
      TF_RETURN_IF_ERROR(c->GetAttr("counts", &counts));
      for (auto cnt : counts) {
        if (cnt < 0) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    " Invalid attr counts");
        }
      }
      int hop_num;
      TF_RETURN_IF_ERROR(c->GetAttr("hop_num", &hop_num));
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((T.size() != 1 && T.size() != 2) ||
          (T.size() == 1 && (T[0] != tensorflow::DT_INT64)) ||
          (T.size() == 2 &&
           (T[0] != tensorflow::DT_INT64 || T[1] != tensorflow::DT_FLOAT))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid input ids");
      }
      int size = 1, cur = 1;
      for (size_t i = 0; i < counts.size(); ++i) {
        cur = cur * counts[i];
        size += cur;
      }
      c->set_output(0, c->MakeShape({c->Dim(ids, 0), size}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectSeqByMultiHop

collect multi hop sequence op

ids: input, vertex id list
metapath: input, edge types tensor list,the size of list is hop num
counts: attr, count list, sample neighbor count every hop
hop_num: attr, hop num
T: attr, result type list [DT_INT64, DT_FLOAT], DT_FLOAT is optional
outs: output,  output list, including ids value

)doc");

REGISTER_OP("CollectSeqByRWWithBias")
    .Input("ids: int64")
    .Input("metapath: walk_length * uint8")
    .Attr("walk_length: int >= 1")
    .Attr("repetition: int >= 1")
    .Attr("p: float = 1.0")
    .Attr("q: float = 1.0")
    .Output("output: int64")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &types));
      int repetition = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("repetition", &repetition));
      int walk_length = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("walk_length", &walk_length));
      float p = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("p", &p));
      float q = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("q", &q));
      tensorflow::shape_inference::DimensionHandle size;
      c->Multiply(c->Dim(ids, 0), repetition, &size);
      c->set_output(0, c->MakeShape({size, walk_length}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectSeqByRWWithBias

collect sequence by random walk with bias

ids: input, vertex id list
metapath: input, edge types tensor list,the size of list is walk_length
walk_length: attr, walk length
repetition: repetition of ids
p:the coefficient of d_tx=0
q: the coefficient of d_tx=2
output: output, sequence list
)doc");

REGISTER_OP("CollectPairByRWWithBias")
    .Input("ids: int64")
    .Input("metapath: walk_length * uint8")
    .Attr("walk_length: int >= 1")
    .Attr("repetition: int >= 1")
    .Attr("p: float = 1.0")
    .Attr("q: float = 1.0")
    .Attr("context_size: int >= 1")
    .Output("output: int64")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      tensorflow::shape_inference::ShapeHandle types;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &types));
      int repetition = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("repetition", &repetition));
      int walk_length = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("walk_length", &walk_length));
      float p = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("p", &p));
      float q = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("q", &q));
      int context_size = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("context_size", &context_size));
      tensorflow::shape_inference::DimensionHandle pair_size;
      c->Multiply(c->Dim(ids, 0), repetition * context_size *
              (2 * walk_length - context_size + 1), &pair_size);
      c->set_output(0, c->MakeShape({pair_size, 2}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectPairByRWWithBias

collect random walk with bias sequence pair

ids: input, vertex id list
metapath: input, edge types tensor list,the size of list is walk_length
walk_length: attr, walk length
repetition: repetition of ids
p:the coefficient of d_tx=0
q: the coefficient of d_tx=2
context_size: the sequence context size that combining in pair
output: output, pairs list
)doc");
