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

#include <errno.h>

#include "filesystem.h"
#include "utils/file_reader.h"
#include "utils/file_writer.h"

#include "glog/logging.h"

namespace galileo {
namespace utils {

struct HdfsConnectParam {
  HdfsConnectParam() : port(0) {}

  std::string nn;
  int port;
};

class HdfsFileSystem : public FileSystem {
 public:
  HdfsFileSystem() = default;
  ~HdfsFileSystem() { this->_disconnect(); }

 public:
  virtual bool Init(const FileConfig& file_config) {
    HdfsConnectParam hdfs_connect_param;
    hdfs_connect_param.nn = file_config.at("hdfs_addr");
    hdfs_connect_param.port = atoi(file_config.at("hdfs_port").c_str());
    return _connect(&hdfs_connect_param);
  }
  virtual std::shared_ptr<galileo::utils::IFileReader> OpenFileReader(
      const char* file_path) {
    HdfsFileReader* hdfs_file_reader = new HdfsFileReader(hdfs_);
    if (!hdfs_file_reader->Open(file_path)) {
      LOG(ERROR) << " Open file(" << file_path << ") failed";
      delete hdfs_file_reader;
      return std::shared_ptr<galileo::utils::IFileReader>(nullptr);
    }
    return std::shared_ptr<galileo::utils::IFileReader>(hdfs_file_reader);
  }
  virtual std::shared_ptr<galileo::utils::IFileWriter> OpenFileWriter(
      const char* file_path) {
    HdfsFileWriter* hdfs_file_writer = new HdfsFileWriter(hdfs_);
    if (!hdfs_file_writer->Open(file_path)) {
      LOG(ERROR) << " Open file(" << file_path << ") failed";
      delete hdfs_file_writer;
      return std::shared_ptr<galileo::utils::IFileWriter>(nullptr);
    }
    return std::shared_ptr<galileo::utils::IFileWriter>(hdfs_file_writer);
  }
  // check file is exist
  virtual bool IsFileExist(const char* path) {
    if (unlikely(NULL == path || path[0] == '\0')) {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);
    if (0 != hdfsExists(hdfs_, path)) {
      return false;
    }

    return true;
  }

  // check folder is exist
  virtual bool IsFolderExist(const char* path) {
    if (unlikely(NULL == path || path[0] == '\0')) {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);
    if (0 != hdfsExists(hdfs_, path)) {
      return false;
    }

    return true;
  }

  virtual bool ListFiles(const char* path, std::vector<std::string>& files) {
    if (unlikely(NULL == path || path[0] == '\0')) {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);

    int count = 0;
    hdfsFileInfo* infos = hdfsListDirectory(hdfs_, path, &count);
    if (unlikely(!infos)) {
      return false;
    }

    files.clear();
    for (int idx = 0; idx < count; ++idx) {
      hdfsFileInfo& info = infos[idx];
      if ((likely(info.mKind == kObjectKindFile)) &&
          (strstr(info.mName, "_SUCCESS") == NULL)) {
        files.push_back(info.mName);
      }
    }

    hdfsFreeFileInfo(infos, count);
    return true;
  }

  virtual bool ListFolders(const char* path,
                           std::vector<std::string>& folders) {
    if (unlikely(NULL == path || path[0] == '\0')) {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);

    int count = 0;
    hdfsFileInfo* infos = hdfsListDirectory(hdfs_, path, &count);
    if (unlikely(!infos)) {
      return false;
    }

    folders.clear();
    for (int idx = 0; idx < count; ++idx) {
      hdfsFileInfo& info = infos[idx];
      if (info.mKind == kObjectKindDirectory) {
        folders.push_back(info.mName);
      }
    }

    hdfsFreeFileInfo(infos, count);
    return true;
  }

  // rm file
  virtual bool RemoveFile(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);

    if (unlikely(0 != hdfsDelete(hdfs_, path, 0))) {
      return false;
    }

    return true;
  }

  // rm folder
  virtual bool RemoveFolder(const char* path) {
    if (NULL == path || path[0] == '\0') {
      LOG(ERROR) << " The path param is null";
      return false;
    }
    assert(NULL != hdfs_);

    if (unlikely(0 != hdfsDelete(hdfs_, path, 1))) {
      return false;
    }

    return true;
  }

  // create one folder recursively
  virtual bool CreateDirRecursion(const char* dir) {
    if (NULL == dir || dir[0] == '\0') {
      LOG(ERROR) << " The dir param is null";
      return false;
    }
    assert(NULL != hdfs_);

    if (unlikely(0 != hdfsCreateDirectory(hdfs_, dir))) {
      return false;
    }

    return true;
  }

 private:
  bool _connect(const HdfsConnectParam* param) {
    this->_disconnect();

    hdfsBuilder* builder = NULL;
    if (param) {
      builder = this->_CreateBuild(*param);
    } else {
      builder = this->_CreateBuild(HdfsConnectParam());
    }

    if (!builder) {
      return false;
    }

    hdfs_ = hdfsBuilderConnect(builder);
    return NULL != hdfs_;
  }
  void _disconnect() {
    if (likely(hdfs_ != NULL)) {
      hdfsDisconnect(hdfs_);
    }
  }

  hdfsBuilder* _CreateBuild(const HdfsConnectParam& param) {
    hdfsBuilder* builder = hdfsNewBuilder();
    if (unlikely(!builder)) {
      return NULL;
    }

    if (param.nn.empty()) {
      hdfsBuilderSetNameNode(builder, "default");
    } else {
      hdfsBuilderSetNameNode(builder, param.nn.c_str());
    }

    if (param.port != 0) {
      hdfsBuilderSetNameNodePort(builder, param.port);
    }

    return builder;
  }

 private:
  hdfsFS hdfs_;
};

}  // namespace utils
}  // namespace galileo
