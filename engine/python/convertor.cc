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
#include <thread>

#include "convertor/tool.h"
#include "convertor/tool_config.h"

namespace py = pybind11;
using ToolConfig = galileo::convertor::ToolConfig;
void StartConvert(ToolConfig& config) { galileo::convertor::Start(config); }

PYBIND11_MODULE(py_convertor, m) {
  py::class_<ToolConfig>(m, "Config")
      .def(py::init<>())
      .def_readwrite("slice_count", &ToolConfig::slice_count)
      .def_readwrite("process_index", &ToolConfig::process_index)
      .def_readwrite("process_count", &ToolConfig::process_count)
      .def_readwrite("worker_count", &ToolConfig::worker_count)
      .def_readwrite("vertex_source_path", &ToolConfig::vertex_source_path)
      .def_readwrite("vertex_binary_path", &ToolConfig::vertex_binary_path)
      .def_readwrite("edge_source_path", &ToolConfig::edge_source_path)
      .def_readwrite("edge_binary_path", &ToolConfig::edge_binary_path)
      .def_readwrite("schema_path", &ToolConfig::schema_path)
      .def_readwrite("field_separator", &ToolConfig::field_separator)
      .def_readwrite("array_separator", &ToolConfig::array_separator)
      .def_readwrite("hdfs_addr", &ToolConfig::hdfs_addr)
      .def_readwrite("hdfs_port", &ToolConfig::hdfs_port)
      .def_readwrite("coordinate_cpu", &ToolConfig::coordinate_cpu);

  m.def("start_convert", &StartConvert, "start convert tool with ToolConfig",
        py::arg("config"));
}
