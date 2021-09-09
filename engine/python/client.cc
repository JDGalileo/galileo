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

#include "client/dgraph_global.h"
#include "client/dgraph_type.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using DGraphConfig = galileo::client::DGraphConfig;
using GraphMeta = galileo::client::GraphMeta;

PYBIND11_MODULE(py_client, m) {
  m.doc() =
      "pybind11 dgraph client module, include CreateDGraph DestroyDGraph "
      "CollectGraphMeta";

  py::class_<DGraphConfig>(m, "DGraphConfig")
      .def(py::init<>())
      .def_readwrite("zk_addr", &DGraphConfig::zk_addr)
      .def_readwrite("zk_path", &DGraphConfig::zk_path)
      .def_readwrite("rpc_timeout_ms", &DGraphConfig::rpc_timeout_ms)
      .def_readwrite("rpc_body_size", &DGraphConfig::rpc_body_size)
      .def_readwrite("rpc_bthread_concurrency",
                     &DGraphConfig::rpc_bthread_concurrency);

  py::class_<GraphMeta>(m, "GraphMeta")
      .def(py::init<>())
      .def_readwrite("vertex_size", &GraphMeta::vertex_size)
      .def_readwrite("edge_size", &GraphMeta::edge_size);

  m.def("CreateDGraph", &galileo::client::CreateDGraph,
        "create the global dgraph instance.");

  m.def("DestroyDGraph", &galileo::client::DestroyDGraph,
        "destroy the global dgraph instance.");

  m.def("CollectGraphMeta", &galileo::client::CollectGraphMeta,
        "get the dgraph meta info.");
}
