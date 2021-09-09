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

#include "convertor/tool.h"
#include "common/macro.h"
#include "convertor/converter.h"
#include "glog/logging.h"

namespace galileo {
namespace convertor {
ToolConfig G_ToolConfig;
bool Start(ToolConfig& tool_config) {
  G_ToolConfig = tool_config;

  galileo::convertor::Converter converter;
  if (unlikely(!converter.Initialize(G_ToolConfig.slice_count,
                                     G_ToolConfig.schema_path.c_str()))) {
    LOG(ERROR) << " converter Initialize fail!";
    return false;
  }
  if (unlikely(!converter.Start(G_ToolConfig.worker_count,
                                G_ToolConfig.vertex_source_path.c_str(),
                                G_ToolConfig.vertex_binary_path.c_str(),
                                G_ToolConfig.edge_source_path.c_str(),
                                G_ToolConfig.edge_binary_path.c_str()))) {
    LOG(ERROR) << " converter start fail!";
    return false;
  }
  return true;
}

}  // namespace convertor
}  // namespace galileo
