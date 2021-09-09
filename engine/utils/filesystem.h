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

#include <memory.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

#include "common/macro.h"
#include "common/types.h"
#include "utils/file_reader.h"
#include "utils/file_writer.h"

namespace galileo {
namespace utils {
using FileConfig = std::unordered_map<std::string, std::string>;

class FileSystem {
 public:
  FileSystem() = default;

  virtual ~FileSystem() = default;

 public:
  virtual bool Init(const FileConfig& file_config) = 0;
  virtual std::shared_ptr<galileo::utils::IFileReader> OpenFileReader(
      const char* file_path) = 0;
  virtual std::shared_ptr<galileo::utils::IFileWriter> OpenFileWriter(
      const char* file_path) = 0;
  // check dir is exist
  virtual bool IsFolderExist(const char* path) = 0;

  // check file is exist
  virtual bool IsFileExist(const char* path) = 0;

  virtual bool ListFiles(const char* dir, std::vector<std::string>& files) = 0;

  virtual bool ListFolders(const char* dir,
                           std::vector<std::string>& floders) = 0;

  // rm file
  virtual bool RemoveFile(const char* path) = 0;

  // rm dir
  virtual bool RemoveFolder(const char* path) = 0;

  // create one folder recursively
  virtual bool CreateDirRecursion(const char* dir) = 0;
};

}  // namespace utils
}  // namespace galileo
