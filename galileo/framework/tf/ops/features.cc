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

REGISTER_OP("CollectFeature")
    .Input("ids: T")
    .Attr("fnames: list(string)")
    .Attr("dimensions: list(int)")
    .Attr("T: list({int64, uint8})")
    .Attr("TO: list(type)")
    .Output("outs: TO")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((T.size() != 1 && T.size() != 3) ||
          (T.size() == 1 && T[0] != tensorflow::DT_INT64) ||
          (T.size() == 3 &&
           (T[0] != tensorflow::DT_INT64 || T[1] != tensorflow::DT_INT64 ||
            T[2] != tensorflow::DT_INT8))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  "Invalid input ids");
      }

      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      std::vector<std::string> fnames;
      TF_RETURN_IF_ERROR(c->GetAttr("fnames", &fnames));
      std::vector<int> dimensions;
      TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
      for (auto& dim : dimensions) {
        if (dim < 0) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    "Invalid attr dimensions");
        }
      }
      tensorflow::DataTypeVector TO;
      TF_RETURN_IF_ERROR(c->GetAttr("TO", &TO));
      if (fnames.size() != dimensions.size() || fnames.size() != TO.size()) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr fnames or dimensions or TO");
      }
      auto dim = c->Dim(ids, 0);
      for (size_t idx = 0; idx < TO.size(); ++idx) {
        c->set_output(idx, c->MakeShape({dim, c->MakeDim(dimensions[idx])}));
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectFeature

collect graph entity feature op

ids: input, ids list , vertex or edge
fnames: attr, feature name list
dimensions: attr, max dimensions for every feature
T: attr, input list type
T: attr output list type
outs: output, output list

)doc");

REGISTER_OP("CollectPodFeature")
    .Input("ids: T")
    .Attr("fnames: list(string)")
    .Attr("dimensions: list(int)")
    .Attr("T: list({int64, uint8})")
    .Attr("N: int")
    .Output("types: int32")
    .Output("outs: N * int8")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((T.size() != 1 && T.size() != 3) ||
          (T.size() == 1 && T[0] != tensorflow::DT_INT64) ||
          (T.size() == 3 &&
           (T[0] != tensorflow::DT_INT64 || T[1] != tensorflow::DT_INT64 ||
            T[2] != tensorflow::DT_INT8))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  "Invalid input ids");
      }

      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      std::vector<std::string> fnames;
      TF_RETURN_IF_ERROR(c->GetAttr("fnames", &fnames));
      std::vector<int> dimensions;
      TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
      for (auto& dim : dimensions) {
        if (dim < 0) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    " Invalid attr dimensions");
        }
      }
      int N = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      if (fnames.size() != dimensions.size() ||
          fnames.size() != static_cast<size_t>(N)) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr fnames or dimensions or N");
      }

      c->set_output(0, c->MakeShape({c->MakeDim(N)}));
      auto ids_dim = c->Dim(ids, 0);
      auto unknow_dim = c->UnknownDim();
      for (int idx = 1; idx < N + 1; ++idx) {
        c->set_output(
            idx, c->MakeShape(
                     {ids_dim, c->MakeDim(dimensions[idx - 1]), unknow_dim}));
      }

      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectPodFeature

collect graph entity feature op

ids: input, ids list , edge or vertex
fnames: attr, feature name list
dimensions: attr, max dimensions for every feature
T: attr, input list type
N: attr, result list len
types: output list really type
outs: output, output int8 list

)doc");

REGISTER_OP("CollectStringFeature")
    .Input("ids: T")
    .Attr("fnames: list(string)")
    .Attr("dimensions: list(int)")
    .Attr("T: list({int64, uint8})")
    .Attr("N: int")
    .Output("outs: N * string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::DataTypeVector T;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &T));
      if ((T.size() != 1 && T.size() != 3) ||
          (T.size() == 1 && T[0] != tensorflow::DT_INT64) ||
          (T.size() == 3 &&
           (T[0] != tensorflow::DT_INT64 || T[1] != tensorflow::DT_INT64 ||
            T[2] != tensorflow::DT_INT8))) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr T");
      }

      tensorflow::shape_inference::ShapeHandle ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ids));
      std::vector<std::string> fnames;
      TF_RETURN_IF_ERROR(c->GetAttr("fnames", &fnames));
      std::vector<int> dimensions;
      TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
      for (auto& dim : dimensions) {
        if (dim < 0) {
          return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    " Invalid attr dimensions");
        }
      }
      int N = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      if (fnames.size() != dimensions.size() ||
          fnames.size() != static_cast<size_t>(N)) {
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  " Invalid attr fnames or dimensions or N");
      }

      auto dim = c->Dim(ids, 0);
      for (size_t idx = 0; idx < T.size(); ++idx) {
        c->set_output(idx, c->MakeShape({dim, c->MakeDim(dimensions[idx])}));
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
CollectStringFeature

collect graph entity feature op

ids: input, ids list , edge or vertex
fnames: input, feature name list
dimensions: attr, max dimensions for every feature
T: attr, input list type info
N: attr, result list len
outs: output, output string list

)doc");
