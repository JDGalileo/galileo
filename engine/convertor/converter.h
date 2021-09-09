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

#include <mutex>

#include "common/macro.h"
#include "common/schema.h"
#include "common/types.h"
#include "utils/buffer.h"
#include "utils/task_thread_pool.h"

#include "edge_worker.h"
#include "file_manager.h"
#include "vertex_worker.h"

namespace galileo {
namespace convertor {

using Buffer = galileo::utils::Buffer;

struct SliceBuffer {
  Buffer buffer;
  std::mutex locker;
};

class Converter {
  friend class Worker;
  friend class VertexWorker;
  friend class EdgeWorker;

 public:
  Converter();
  ~Converter();

 public:
  bool Initialize(int slice_count, const char* shcema_path);
  bool Start(int worker_count, const char* vertex_source_path,
             const char* vertex_binary_path, const char* edge_source_path,
             const char* edge_binary_path);

 private:
  bool _StartVertexWorkerProcess(const char* v_source_path,
                                 const char* v_binary_path, int worker_count);

  bool _StartEdgeWorkerProcess(const char* e_source_path,
                               const char* e_binary_path, int worker_count);

 private:
  galileo::schema::Schema schema_;

  SliceBuffer* slices_;
  int slice_count_;
  FileManager file_manager_;
  galileo::utils::TaskThreadPool task_thread_pool_;
};

}  // namespace convertor
}  // namespace galileo
