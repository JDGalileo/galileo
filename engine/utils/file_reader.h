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
#include "glog/logging.h"
#include "libhdfs/hdfs.h"

namespace galileo {
namespace utils {

class IFileReader {
 public:
  virtual ~IFileReader() {}
  virtual size_t Read(char* buf, size_t len, bool* eof) = 0;
  virtual void Close() = 0;
  virtual bool Open(const char* filepath) = 0;
  virtual std::string& GetFileName() = 0;
};

class HdfsFileReader : public IFileReader {
 protected:
  hdfsFS fs_;
  hdfsFile fp_;
  std::string file_name_;

 public:
  HdfsFileReader(hdfsFS fs) : fs_(fs), fp_(nullptr) {}
  virtual ~HdfsFileReader() {
    this->Close();
    fs_ = NULL;
  }

  virtual bool Open(const char* filepath) {
    if (fp_) {
      this->Close();
    }

    file_name_ = filepath;
    fp_ = hdfsOpenFile(fs_, filepath, O_RDONLY, 0, 0, 0);
    return nullptr != fp_;
  }

  virtual size_t Read(char* buf, size_t len, bool* eof) {
    tSize ret = hdfsRead(fs_, fp_, buf, len);
    if (ret == -1) {
      LOG(ERROR) << "read file(" << GetFileName() << ") failed!";
      if (eof) {
        *eof = true;
      }
      return 0;
    }
    size_t readed = (size_t)ret;
    if (readed < len) {
      if (eof) {
        *eof = (readed == 0);
      }
    }
    return readed;
  }
  virtual void Close() {
    if (fp_) {
      hdfsCloseFile(fs_, fp_);
      fp_ = NULL;
    }
  }
  virtual std::string& GetFileName() { return file_name_; }
};

class LocalFileReader : public IFileReader {
 protected:
  FILE* fp_;
  std::string file_name_;

 public:
  LocalFileReader() : fp_(nullptr) {}

  LocalFileReader(LocalFileReader&& localFileReader) {
    this->fp_ = localFileReader.fp_;
    localFileReader.fp_ = NULL;
    file_name_ = localFileReader.file_name_;
  }
  ~LocalFileReader() { this->Close(); }

 public:
  bool Open(const char* filepath) {
    if (fp_) {
      this->Close();
    }

    fp_ = fopen(filepath, "rb");
    if (!fp_) {
      return false;
    }
    file_name_ = filepath;
    return true;
  }

  size_t Read(char* buf, size_t len, bool* eof) {
    size_t readed = fread(buf, 1, len, fp_);
    if (readed < len) {
      if (eof) {
        *eof = (readed == 0);
      }
    }

    return readed;
  }

  void Close() {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }
  std::string& GetFileName() { return file_name_; }
};

}  // namespace utils
}  // namespace galileo
