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

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("collect_entity", &torch::glo::CollectEntity, "collect entity (CPU)");

  m.def("collect_state_neighbor", &torch::glo::CollectStateNeighbor,
        "collect state neighbor (CPU)");
  m.def("collect_neighbor", &torch::glo::CollectNeighbor,
        "collect neighbor (CPU)");

  m.def("collect_pod_feature", &torch::glo::CollectPodFeature,
        "collect pod neighbor (CPU)");
  m.def("collect_string_feature", &torch::glo::CollectStringFeature,
        "collect string feature (CPU)");

  m.def("collect_seq_by_multi_hop", &torch::glo::CollectSeqByMultiHop,
        "collect seq by multi hop (CPU)");
  m.def("collect_seq_by_rw_with_bias", &torch::glo::CollectSeqByRWWithBias,
        py::arg("ids"), py::arg("metapath"), py::arg("repetition"),
        py::arg("p") = 1.0, py::arg("q") = 1.0,
        "collect seq by rw with bias (CPU)");
  m.def("collect_pair_by_rw_with_bias", &torch::glo::CollectPairByRWWithBias,
        py::arg("ids"), py::arg("metapath"), py::arg("repetition"),
        py::arg("context_size"), py::arg("p") = 1.0, py::arg("q") = 1.0,
        "collect pair by rw with bias (CPU)");
}
