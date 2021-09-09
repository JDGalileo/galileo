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

#include <string>
#include <vector>
#include "convertor/transform_help.h"
#include "utils/buffer.h"
#include "utils/task_thread_pool.h"

namespace galileo {
namespace convertor {

using Buffer = galileo::utils::Buffer;

class FileSystem;
class Converter;

class Worker {
 public:
  Worker() : success_(false), converter_(NULL) {}

  virtual ~Worker() = default;

 public:
  void Init(galileo::utils::TaskThreadPool* task_thread_pool,
            Converter* converter) noexcept;
  bool operator()(const char* line, size_t len);
  bool IsSuccess() { return success_; }

 public:
  virtual bool ParseRecord(std::vector<std::vector<char*>>& fields) = 0;
  virtual size_t AllocNextId() = 0;

 protected:
  bool WriteRecord(int slice_id, Record& record);

 protected:
  bool success_;
  Converter* converter_;

  Record record_;
};

}  // namespace convertor
}  // namespace galileo
