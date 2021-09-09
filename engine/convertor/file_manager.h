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

#include <stdio.h>
#include <memory>
#include <string>

#include "utils/file_reader.h"
#include "utils/file_writer.h"
#include "utils/filesystem.h"

#include "common/macro.h"
#include "common/types.h"
#include "convertor/tool_config.h"

namespace galileo {
namespace convertor {

class Worker;

class FileManager {
 public:
  FileManager();

  ~FileManager();

 public:
  bool OpenFilesystem();
  int Initialize(const char* source_file, const char* binary_path,
                 const int slice, const char* prefix);
  std::shared_ptr<galileo::utils::IFileReader> OpenFileReader(
      const char* file_path) {
    return file_system_->OpenFileReader(file_path);
  }
  bool read(const size_t file_idx, Worker* worker);
  bool WriteSlice(const size_t slice_id, const char* buffer, size_t len);
  size_t GetFileNum() const { return readers_.size(); };
  void Restart();

 protected:
  bool _SpliceSliceFilename(const char* path, const int slice,
                            const char* prefix,
                            std::vector<std::string>& files);
  std::vector<std::string> _SpliceFiles(const std::vector<std::string>& files);

 private:
  galileo::utils::FileSystem* file_system_;
  std::vector<std::shared_ptr<galileo::utils::IFileReader>> readers_;
  std::vector<std::shared_ptr<galileo::utils::IFileWriter>> writers_;
};

}  // namespace convertor
}  // namespace galileo
