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

#include <pybind11/pybind11.h>

#include "service/config.h"
#include "service/service.h"

namespace py = pybind11;
using GraphConfig = galileo::service::GraphConfig;

PYBIND11_MODULE(py_service, m) {
  py::class_<GraphConfig>(m, "Config")
      .def(py::init<>())
      .def_readwrite("shard_index", &GraphConfig::shard_index)
      .def_readwrite("shard_count", &GraphConfig::shard_count)
      .def_readwrite("thread_num", &GraphConfig::thread_num)
      .def_readwrite("hdfs_addr", &GraphConfig::hdfs_addr)
      .def_readwrite("hdfs_port", &GraphConfig::hdfs_port)
      .def_readwrite("schema_path", &GraphConfig::schema_path)
      .def_readwrite("data_path", &GraphConfig::data_path)
      .def_readwrite("zk_addr", &GraphConfig::zk_addr)
      .def_readwrite("zk_path", &GraphConfig::zk_path);

  m.def("start", &galileo::service::StartService,
        "start graph server with GraphConfig struct and port",
        py::arg("config"), py::arg("port") = 0, py::arg("daemon") = false);
}
