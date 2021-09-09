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

#include "worker.h"
#include <unistd.h>
#include <cmath>
#include "converter.h"
#include "utils/buffer.h"
#include "utils/string_util.h"
#include "utils/task_thread_pool.h"

#include "common/macro.h"
#include "convertor/file_manager.h"
#include "convertor/transform_help.h"

#include "glog/logging.h"

namespace galileo {
namespace convertor {

void Worker::Init(galileo::utils::TaskThreadPool* task_thread_pool,
                  Converter* converter) noexcept {
  converter_ = converter;
  task_thread_pool->AddTask([this, converter]() {
    while (1) {
      size_t idx = this->AllocNextId();
      if (idx >= converter_->file_manager_.GetFileNum()) {
        success_ = true;
        break;
      }
      if (unlikely(!converter_->file_manager_.read(idx, this))) {
        LOG(ERROR) << "parse file error:" << idx;
        break;
      }
    }
  });
}

bool Worker::operator()(const char* line, size_t len) {
  if (G_ToolConfig.coordinate_cpu > 0) {
    usleep(G_ToolConfig.coordinate_cpu);
  }

  const char* field_split = G_ToolConfig.field_separator.c_str();
  const char* array_split = G_ToolConfig.array_separator.c_str();
  assert(1 == strlen(field_split));
  assert(1 == strlen(array_split));
  char f_split = field_split[0];
  char a_split = array_split[0];
  std::vector<std::vector<char*>> fields;
  TransformHelp::SplitLine((char*)line, len, f_split, a_split, fields);
  return this->ParseRecord(fields);
}

bool Worker::WriteRecord(int slice_id, Record& record) {
  SliceBuffer& slice = converter_->slices_[slice_id];
  FileManager* file_manager = &converter_->file_manager_;
  std::lock_guard<std::mutex> auto_lock(slice.locker);
  size_t avail = slice.buffer.avail();
  size_t fix_size = record.fix_fields.size();
  size_t vary_size = record.vary_fields.size();
  size_t vary_state_size = record.vary_state.size();
  size_t record_size = fix_size + vary_state_size + vary_size;
  if (avail < record_size + 8) {
    file_manager->WriteSlice(slice_id, slice.buffer.buffer(),
                             slice.buffer.size());
    slice.buffer.clear();
  }
  assert(slice.buffer.avail() >= record_size + 8);
  if (slice.buffer.avail() < record_size + 8) {
    LOG(ERROR) << " not enough space for write record";
    return false;
  }
  if (unlikely(!slice.buffer.write((char*)&record_size, sizeof(uint16_t)))) {
    LOG(ERROR) << " write record_size value fail!";
    return false;
  }
  if (unlikely(!slice.buffer.write(record.fix_fields.buffer(), fix_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(
          !slice.buffer.write(record.vary_state.buffer(), vary_state_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(!slice.buffer.write(record.vary_fields.buffer(), vary_size))) {
    LOG(ERROR) << " write record vary field content fail!";
    return false;
  }

  return true;
}
}  // namespace convertor
}  // namespace galileo
