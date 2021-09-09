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

#include "glog/logging.h"

namespace galileo {
namespace client {

DGraph *gDGraph = nullptr;

bool CreateDGraph(const DGraphConfig &config) {
  if (nullptr == gDGraph) {
    gDGraph = new DGraph();
    if (!gDGraph->Initialize(config)) {
      LOG(ERROR) << "Initialize graph without candidate failed.";
      delete gDGraph;
      gDGraph = nullptr;
      return false;
    }
    LOG(INFO) << "Create dgraph instance success.";
  }
  return true;
}

void DestroyDGraph() {
  if (gDGraph != nullptr) {
    delete gDGraph;
    gDGraph = nullptr;
  }
}

GraphMeta CollectGraphMeta() {
  GraphMeta meta_info{0, 0};
  if (nullptr == gDGraph) {
    LOG(ERROR) << "The gDGraph is nullptr!";
    return meta_info;
  }
  if (!gDGraph->CollectGraphMeta(&meta_info)) {
    LOG(ERROR) << "Get meta info fail!";
  }
  return meta_info;
}

}  // namespace client
}  // namespace galileo
