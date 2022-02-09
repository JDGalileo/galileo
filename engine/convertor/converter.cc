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

#include "converter.h"

#include "common/types.h"

#include "convertor/tool_config.h"
#include "glog/logging.h"
#include "quickjson/value.h"

namespace galileo {
namespace convertor {

Converter::Converter() : slices_(NULL), slice_count_(0) {}

Converter::~Converter() {
  if (likely(slices_ != NULL)) {
    delete[] slices_;
    slices_ = NULL;
  }
}

bool Converter::Initialize(int slice_count, const char* schema_path) {
  if (!file_manager_.OpenFilesystem()) {
    return false;
  }

  std::shared_ptr<galileo::utils::IFileReader> file_reader =
      file_manager_.OpenFileReader(schema_path);
  if (nullptr == file_reader.get()) {
    LOG(ERROR) << "read file (" << schema_path << ") failed";
    return false;
  }
  char buff[64 * 1024] = {0};
  bool is_end = false;
  file_reader->Read(buff, 64 * 1024, &is_end);
  if (unlikely(!schema_.Build(buff))) {
    LOG(ERROR) << "build schema fail!";
    return false;
  }

  slices_ = new SliceBuffer[slice_count];
  slice_count_ = slice_count;
  for (int idx = 0; idx < slice_count; ++idx) {
    slices_[idx].buffer.reserve(galileo::common::WRITE_BUFFER_SIZE);
  }

  return true;
}

bool Converter::Start(int worker_count, const char* vertex_source_path,
                      const char* vertex_binary_path,
                      const char* edge_source_path,
                      const char* edge_binary_path) {
  worker_count = worker_count > galileo::common::MAX_THRAED_NUM
                     ? galileo::common::MAX_THRAED_NUM
                     : worker_count;

  LOG(INFO) << " Starting process vertices";
  if (unlikely(!this->_StartVertexWorkerProcess(
          vertex_source_path, vertex_binary_path, worker_count))) {
    LOG(ERROR) << " start vertex worker process fail!";
    return false;
  }

  file_manager_.Restart();

  LOG(INFO) << " Starting process edges";
  if (unlikely(!this->_StartEdgeWorkerProcess(
          edge_source_path, edge_binary_path, worker_count))) {
    LOG(ERROR) << " start edge worker process fail!";
    return false;
  }
  LOG(INFO) << " All complete!";
  return true;
}

#define START_WORKER_PROCESS(SOURCE_FILE, BINARY_PATH, WORKER, WORKER_COUNT,   \
                             PREFIX)                                           \
  {                                                                            \
    int ret = file_manager_.Initialize(SOURCE_FILE, BINARY_PATH, slice_count_, \
                                       PREFIX);                                \
    if (ret < 0) {                                                             \
      LOG(ERROR) << "filesystem initalize fail!";                              \
      return false;                                                            \
    }                                                                          \
    if (ret > 0) {                                                             \
      return true;                                                             \
    }                                                                          \
    WORKER* workers = new WORKER[WORKER_COUNT];                                \
    for (int idx = 0; idx < WORKER_COUNT; ++idx) {                             \
      WORKER& worker = workers[idx];                                           \
      worker.Init(&task_thread_pool_, this);                                   \
    }                                                                          \
    task_thread_pool_.Start(WORKER_COUNT);                                     \
    task_thread_pool_.ShutDown();                                              \
    for (int i = 0; i < WORKER_COUNT; ++i) {                                   \
      if (unlikely(!workers[i].IsSuccess())) {                                 \
        LOG(ERROR) << "worker [" << i << "]fail!";                             \
        delete[] workers;                                                      \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    for (int idx = 0; idx < slice_count_; ++idx) {                             \
      if (slices_[idx].buffer.size() != 0) {                                   \
        if (unlikely(!file_manager_.WriteSlice(idx,                            \
                                               slices_[idx].buffer.buffer(),   \
                                               slices_[idx].buffer.size()))) { \
          LOG(ERROR) << "write slice buffer fail!";                            \
          delete[] workers;                                                    \
          return false;                                                        \
        }                                                                      \
        slices_[idx].buffer.clear();                                           \
      }                                                                        \
    }                                                                          \
    delete[] workers;                                                          \
    return true;                                                               \
  }

bool Converter::_StartVertexWorkerProcess(const char* v_source_path,
                                          const char* v_binary_path,
                                          int worker_count) {
  if (v_source_path == NULL || strlen(v_source_path) <= 0) {
    return true;
  }
  START_WORKER_PROCESS(v_source_path, v_binary_path, VertexWorker, worker_count,
                       "vertex");
}

bool Converter::_StartEdgeWorkerProcess(const char* e_source_path,
                                        const char* e_binary_path,
                                        int worker_count) {
  if (e_source_path == NULL || strlen(e_source_path) <= 0) {
    return true;
  }
  START_WORKER_PROCESS(e_source_path, e_binary_path, EdgeWorker, worker_count,
                       "edge");
}

}  // namespace convertor
}  // namespace galileo
