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

#include "service/graph_loader.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/macro.h"
#include "common/singleton.h"
#include "glog/logging.h"
#include "quickjson/value.h"
#include "utils/hdfs_filesystem.h"
#include "utils/local_filesystem.h"
#include "utils/task_thread_pool.h"

namespace galileo {
namespace service {

GraphLoader::GraphLoader(const galileo::service::GraphConfig& config)
    : graph_config_(config) {}

bool GraphLoader::Init() {
  galileo::utils::FileConfig file_config;
  if (graph_config_.IsLocal()) {
    file_system_.reset(new galileo::utils::LocalFileSystem());
  } else {
    file_config["hdfs_addr"] = graph_config_.hdfs_addr;
    file_config["hdfs_port"] = std::to_string(graph_config_.hdfs_port);
    file_system_.reset(new galileo::utils::HdfsFileSystem());
  }
  if (!file_system_->Init(file_config)) {
    LOG(ERROR) << "[Engine] Init filesystem error";
    return false;
  }
  return true;
}

bool GraphLoader::LoadSchema() {
  std::shared_ptr<galileo::utils::IFileReader> schema_file =
      file_system_->OpenFileReader(graph_config_.schema_path.c_str());
  if (!schema_file) {
    LOG(ERROR) << "[Engine] Can't find the schema file";
    return false;
  }
  const size_t buff_size = 64 * 1024;
  char buff[buff_size] = {0};
  schema_file->Read(buff, buff_size, nullptr);

  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  if (unlikely(!schema->Build(buff))) {
    LOG(ERROR) << "[Engine] Build schema fail!";
    return false;
  }
  return true;
}

bool GraphLoader::_GetPartitionFiles(std::vector<std::string>& vertex_file_list,
                                     std::vector<std::string>& edge_file_list,
                                     size_t& num_partitions) {
  vertex_file_list.clear();
  edge_file_list.clear();

  std::vector<std::string> all_files;
  file_system_->ListFiles(graph_config_.data_path.c_str(), all_files);
  std::ostringstream err_msg;
  err_msg << " NO valid vertex/edge files in path: " << graph_config_.data_path
          << ". vertex pattern: vertex_#_#.dat, edge pattern: edge_#_#.dat";
  if (all_files.empty()) {
    LOG(ERROR) << err_msg.str();
    return false;
  }
  std::unordered_set<int32_t> vertex_part, edge_part;
  for (const auto& path_name : all_files) {
    auto pos = path_name.rfind('/');
    std::string file_name;
    if (pos != std::string::npos) {
      file_name = path_name.substr(pos + 1);
    } else {
      file_name = path_name;
    }
    std::vector<std::string> file_name_vec;
    size_t cnt = galileo::utils::split_string(file_name, '.', &file_name_vec);
    if (cnt != 2 || file_name_vec.back() != "dat") continue;
    cnt = galileo::utils::split_string(file_name, '_', &file_name_vec);
    if (cnt < 3) continue;
    if (file_name_vec[0] != "vertex" && file_name_vec[0] != "edge") continue;
    int32_t part_index;
    try {
      part_index = std::stoi(file_name_vec[1]);
    } catch (const std::invalid_argument&) {
      continue;
    }
    if (file_name_vec[0] == "vertex") {
      vertex_part.insert(part_index);
    } else if (file_name_vec[0] == "edge") {
      edge_part.insert(part_index);
    }
    if (part_index % graph_config_.shard_count != graph_config_.shard_index)
      continue;
    if (file_name_vec[0] == "vertex") {
      vertex_file_list.push_back(path_name);
    } else if (file_name_vec[0] == "edge") {
      edge_file_list.push_back(path_name);
    }
  }
  if (vertex_file_list.empty() && edge_file_list.empty()) {
    LOG(ERROR) << err_msg.str() << " for engine index "
               << graph_config_.shard_index;
    return false;
  }
  if (vertex_part.empty()) {
    LOG(ERROR) << " No vertex partition files";
    return false;
  }
  if (vertex_part.size() != edge_part.size()) {
    LOG(WARNING) << " The number of partition of vertex files and edge files"
                 << " is not same (" << vertex_part.size() << " vs "
                 << edge_part.size() << ")";
  }
  num_partitions = vertex_part.size();
  LOG(INFO) << " Engine index " << graph_config_.shard_index
            << ", vertex files: " << vertex_file_list.size()
            << ", edge files: " << edge_file_list.size()
            << ", num_partitions: " << num_partitions;
  return true;
}

#define RELEASE_VEC(VEC, nore)                                \
  {                                                           \
    if (!VEC.empty()) {                                       \
      LOG(ERROR) << std::to_string(VEC.size()) << " " << nore \
                 << " have been ignored!";                    \
    }                                                         \
    while (!VEC.empty()) {                                    \
      auto it = VEC.end();                                    \
      VEC.erase(--it);                                        \
      if (*it) delete *it;                                    \
    }                                                         \
    VEC.clear();                                              \
  }

bool GraphLoader::LoadGraph(Graph* graph) {
  size_t thread_num = static_cast<size_t>(std::thread::hardware_concurrency());
  std::vector<LoadInfo> load_infos;
  std::vector<std::string> vertex_files, edge_files;
  size_t num_partitions = 0;
  if (!_GetPartitionFiles(vertex_files, edge_files, num_partitions)) {
    return false;
  }
  graph->SetPartitions((uint32_t)num_partitions);

  for (size_t i = 0; i < vertex_files.size(); ++i) {
    load_infos.emplace_back();
    auto& load_info = load_infos.back();
    load_info.file = &vertex_files[i];
    load_info.type = LoadVertexType;
  }
  for (size_t i = 0; i < edge_files.size(); ++i) {
    load_infos.emplace_back();
    auto& load_info = load_infos.back();
    load_info.file = &edge_files[i];
    load_info.type = LoadEdgeType;
  }
  size_t task_num = load_infos.size();
  thread_num = std::min(thread_num, task_num);

  galileo::utils::TaskThreadPool thread_pool;
  std::atomic<bool> success(true);
  for (auto& load_info : load_infos) {
    thread_pool.AddTask([this, &load_info, &success]() {
      if (success && !_LoadData(&load_info)) {
        success = false;
      }
    });
  }
  thread_pool.Start(thread_num);
  thread_pool.ShutDown();
  if (!success) {
    for (size_t i = 0; i < load_infos.size(); i++) {
      const auto& load_info = load_infos.data() + i;
      RELEASE_VEC(load_info->vertex_vec, "vertices");
      RELEASE_VEC(load_info->edge_vec, "edges");
    }
    LOG(ERROR) << " Load graph data failed!";
    return false;
  }
  LOG(INFO) << " Load graph data done";
  // add vertex
  bool stat = true;
  for (size_t i = 0; i < load_infos.size(); i++) {
    const auto& load_info = load_infos.data() + i;
    if (!stat) {
      RELEASE_VEC(load_info->vertex_vec, "vertices");
    } else {
      stat = graph->AddVertexs(load_info->vertex_vec);
      load_info->vertex_vec.clear();
    }
  }
  if (!stat) {
    LOG(ERROR) << " Parse vertex data failed!";
    return false;
  }
  // add edge, relate edge to out point
  for (auto& load_info : load_infos) {
    graph->AddEdges(load_info.edge_vec);
    load_info.edge_vec.clear();
  }

  thread_pool.Start(thread_num);
  thread_pool.ShutDown();
  load_infos.clear();

  if (!graph->BuildGlobalVertexSampler()) {
    LOG(ERROR) << " Build global vertex sampler failed!";
    return false;
  }

  if (!graph->BuildGlobalEdgeSampler()) {
    LOG(ERROR) << " Build global edge sampler failed!";
    return false;
  }

  if (!graph->BuildSubEdgeSampler()) {
    LOG(ERROR) << " Build sub edge sampler failed!";
    return false;
  }

  LOG(INFO) << " Build graph success!"
            << " vertex count: " << graph->GetVertexCount()
            << " ,edge count: " << graph->GetEdgeCount();
  return true;
}

bool GraphLoader::_LoadData(LoadInfo* load_info) {
  auto& file = *load_info->file;
  std::shared_ptr<galileo::utils::IFileReader> reader =
      file_system_->OpenFileReader(file.c_str());
  if (reader.get() == nullptr) {
    LOG(ERROR) << " Open file " << file << " failed!!";
    return false;
  }
  FileReaderHelper file_reader_helper(reader);
  bool success = false;
  switch (load_info->type) {
    case LoadVertexType:
      success = _ParseVertices(file_reader_helper, &load_info->vertex_vec);
      break;
    case LoadEdgeType:
      success = _ParseEdges(file_reader_helper, &load_info->edge_vec);
      break;
    default:
      assert(false);
  }
  if (!success) {
    LOG(ERROR) << file << " Parse error!";
    return false;
  }
  LOG(INFO) << " Load data file done: " << file;

  return true;
}

bool GraphLoader::_ParseVertices(FileReaderHelper& file_reader_helper,
                                 VERTEXVEC* vertex_vec) {
  std::string vertex_info;
  while (file_reader_helper.Read(&vertex_info)) {
    uint8_t type = *reinterpret_cast<const uint8_t*>(vertex_info.c_str());
    Vertex* vertex = new Vertex(type);
    if (!vertex->DeSerialize(type, vertex_info)) {
      return false;
    }
    vertex_vec->emplace_back(vertex);
  }
  return true;
}
bool GraphLoader::_ParseEdges(FileReaderHelper& file_reader_helper,
                              EDGEVEC* edge_vec) {
  std::string edge_info;
  while (file_reader_helper.Read(&edge_info)) {
    uint8_t type = *reinterpret_cast<const uint8_t*>(edge_info.c_str());
    Edge* edge = new Edge(type);
    if (!edge->DeSerialize(type, edge_info)) {
      return false;
    }
    edge_vec->emplace_back(edge);
  }
  return true;
}

}  // namespace service
}  // namespace galileo
