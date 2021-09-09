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

#include "thirdparty/libhdfs/hdfs.h"

namespace galileo {
namespace utils {

class IFileWriter {
 public:
  virtual ~IFileWriter() {}
  virtual bool Write(const char* buffer, size_t len) = 0;
  virtual void Close() = 0;
  virtual bool Open(const char* filepath) = 0;
};

//////////////////////////////////////////////////////////////////
// LocalFileWriter
//////////////////////////////////////////////////////////////////
class LocalFileWriter : public IFileWriter {
 public:
  LocalFileWriter() : file_(NULL) {}

  virtual ~LocalFileWriter() { this->Close(); }

 public:
  virtual bool Open(const char* path) {
    if (file_ != NULL) {
      this->Close();
    }

    file_ = fopen(path, "w");
    return file_ != NULL;
  }

  virtual bool Write(const char* buffer, size_t len) {
    if (file_ == NULL) {
      return false;
    }

    return fwrite(buffer, 1, len, file_) == len;
  }

  bool flush() {
    if (file_ == NULL) return false;

    return fflush(file_) == 0;
  }

  virtual void Close() {
    if (nullptr == file_) return;

    fclose(file_);
    file_ = NULL;
  }

  bool IsOpened() { return file_ != nullptr; }

 private:
  FILE* file_;
};

//////////////////////////////////////////////////////////////////
// HdfsFileWriter
//////////////////////////////////////////////////////////////////
class HdfsFileWriter : public IFileWriter {
 protected:
  hdfsFS fs_;
  hdfsFile fp_;

 public:
  HdfsFileWriter(hdfsFS fs = NULL) : fs_(fs), fp_(nullptr) {}

  ~HdfsFileWriter() {
    this->Close();
    fs_ = NULL;
  }

  virtual bool Open(const char* path) {
    this->Close();

    if (nullptr == fs_) return false;

    fp_ = hdfsOpenFile(fs_, path, O_WRONLY, 0, 0, 0);
    return nullptr != fp_;
  }

  virtual bool Write(const char* buf, size_t len) {
    return hdfsWrite(fs_, fp_, buf, len) != -1;
  }

  bool Flush() { return hdfsFlush(fs_, fp_) == 0; }

  void Close() {
    if (fp_) {
      hdfsCloseFile(fs_, fp_);
      fp_ = NULL;
    }
  }
};

}  // namespace utils
}  // namespace galileo
