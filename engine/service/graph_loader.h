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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/schema.h"
#include "common/types.h"
#include "service/edge.h"
#include "service/file_reader_helper.h"
#include "service/graph.h"
#include "service/vertex.h"
#include "utils/filesystem.h"

namespace galileo {
namespace service {

enum LoadType { LoadVertexType, LoadEdgeType };

using VERTEXVEC = std::vector<Vertex*>;
using EDGEVEC = std::vector<Edge*>;
using Schema = galileo::schema::Schema;

struct LoadInfo {
  std::string* file;
  LoadType type;
  VERTEXVEC vertex_vec;
  EDGEVEC edge_vec;
};

class GraphLoader {
 public:
  GraphLoader(const galileo::service::GraphConfig& config);
  virtual ~GraphLoader() {}

  bool Init();
  bool LoadSchema();
  bool LoadGraph(Graph* graph);

 private:
  bool _LoadData(LoadInfo* load_info);

  bool _ParseVertices(FileReaderHelper& file_reader_helper,
                      VERTEXVEC* vertex_vec);

  bool _ParseEdges(FileReaderHelper& file_reader_helper, EDGEVEC* edge_vec);

  bool _GetPartitionFiles(std::vector<std::string>& vertex_file_list,
                          std::vector<std::string>& edge_file_list,
                          size_t& num_partitions);

 private:
  std::shared_ptr<galileo::utils::FileSystem> file_system_;
  galileo::service::GraphConfig graph_config_;
};
}  // namespace service
}  // namespace galileo
