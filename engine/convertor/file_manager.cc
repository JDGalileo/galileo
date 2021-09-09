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

#include "file_manager.h"
#include "glog/logging.h"
#include "text_reader.h"
#include "utils/hdfs_filesystem.h"
#include "utils/local_filesystem.h"

namespace galileo {
namespace convertor {

FileManager::FileManager() {}
FileManager::~FileManager() {
  readers_.clear();
  writers_.clear();
}
bool FileManager::OpenFilesystem() {
  galileo::utils::FileConfig file_config;
  if (!G_ToolConfig.IsLocal()) {
    file_system_ = new galileo::utils::HdfsFileSystem();
    file_config["hdfs_addr"] = G_ToolConfig.hdfs_addr;
    file_config["hdfs_port"] = std::to_string(G_ToolConfig.hdfs_port);
    if (!file_system_->Init(file_config)) {
      LOG(ERROR) << "connect hdfs filesystem fail!";
      return false;
    }
  } else {
    file_system_ = new galileo::utils::LocalFileSystem();
  }
  return true;
}

int FileManager::Initialize(const char* source_file, const char* binary_path,
                            const int slice, const char* prefix) {
  std::vector<std::string> source_files;
  file_system_->ListFiles(source_file, source_files);
  if (source_files.size() <= 0) {
    LOG(ERROR) << " No source files to process in "<< source_file;
    return 1;
  }
  source_files = this->_SpliceFiles(source_files);
  if (source_files.size() <= 0) {
    LOG(ERROR) << " No source files to process for worker index "
      << G_ToolConfig.process_index;
    return 1;
  }
  for (auto read_iter = source_files.begin(); read_iter != source_files.end();
       ++read_iter) {
    std::shared_ptr<galileo::utils::IFileReader> file_reader =
        file_system_->OpenFileReader(read_iter->c_str());
    if (nullptr == file_reader.get()) {
      LOG(ERROR) << " open the source file fail!"
                 << " filepath:" << *read_iter << " errno:" << errno;
      return -1;
    }
    readers_.push_back(file_reader);
  }

  std::vector<std::string> binary_files;
  this->_SpliceSliceFilename(binary_path, slice, prefix, binary_files);

  for (auto write_iter = binary_files.begin(); write_iter != binary_files.end();
       ++write_iter) {
    std::shared_ptr<galileo::utils::IFileWriter> file_writer =
        file_system_->OpenFileWriter(write_iter->c_str());
    if (nullptr == file_writer.get()) {
      LOG(ERROR) << " open the binary file fail!"
                 << " filepath:" << *write_iter << " errno:" << errno;
      return -1;
    }
    writers_.push_back(file_writer);
  }

  return 0;
}

void FileManager::Restart() {
  readers_.clear();
  writers_.clear();
}

bool FileManager::read(const size_t file_idx, Worker* worker) {
  assert(file_idx < readers_.size());
  galileo::convertor::TextReader reader(readers_[file_idx].get());
  if (!reader.Load(*worker)) {
    LOG(ERROR) << "source file error :"
               << readers_[file_idx]->GetFileName().c_str();
    return false;
  }
  return true;
}

bool FileManager::WriteSlice(const size_t slice_id, const char* buffer,
                             size_t len) {
  assert(slice_id < writers_.size());
  return writers_[slice_id]->Write(buffer, len);
}

bool FileManager::_SpliceSliceFilename(const char* path, const int slice,
                                       const char* prefix,
                                       std::vector<std::string>& files) {
  if (unlikely(!file_system_->IsFolderExist(path))) {
    if (unlikely(!file_system_->CreateDirRecursion(path))) {
      return false;
    }
  }

  char filepath[galileo::common::MAX_PATH_LEN];
  for (int idx = 0; idx < slice; ++idx) {
    memset(filepath, 0, galileo::common::MAX_PATH_LEN);
    snprintf(filepath, 128, "%s/%s_%d_%d.dat", path, prefix, idx,
             G_ToolConfig.process_index);
    files.emplace_back(filepath);
  }

  return true;
}


std::vector<std::string> FileManager::_SpliceFiles(const
    std::vector<std::string>& files) {
  size_t all_files_count = files.size();
  size_t files_count = all_files_count / G_ToolConfig.process_count;
  size_t remainder = all_files_count % G_ToolConfig.process_count;
  size_t begin_pos = G_ToolConfig.process_index * files_count;
  if ( remainder > 0 ) {
    if ( (size_t)G_ToolConfig.process_index < remainder) {
      // add one file to first `remainder` index
      ++files_count;
      begin_pos += G_ToolConfig.process_index;
    } else {
      begin_pos += remainder;
    }
  }
  if (begin_pos >= all_files_count) {
    return std::vector<std::string>();
  }
  size_t end_pos = begin_pos + files_count;
  return std::vector<std::string>(files.begin() + begin_pos,
                                  files.begin() + end_pos);
}

}  // namespace convertor
}  // namespace galileo
