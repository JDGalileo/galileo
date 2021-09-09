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
#include <memory>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "utils/file_reader.h"

#define BUFFER_SIZE 4096
namespace galileo {
namespace service {

class FileReaderHelper {
 public:
  FileReaderHelper(std::shared_ptr<galileo::utils::IFileReader> file_reader)
      : file_reader_(file_reader), read_pos_(0), capacity_(0) {
    buffer_ = malloc(BUFFER_SIZE);
  }
  ~FileReaderHelper() {
    if (nullptr != buffer_) free(buffer_);
  };

  template <typename T>
  inline bool Read(T* data) {
    return ReadData((char*)data, sizeof(T));
  }
  bool Read(std::string* data) {
    uint16_t len = 0;
    bool succ = Read(&len);
    if (succ) {
      data->resize(len);
      succ = ReadData(&(data->front()), len);
    }
    return succ;
  }
  bool ReadData(char* data, size_t len) {
    char* dst = data;
    char* src = static_cast<char*>(buffer_) + read_pos_;
    while (read_pos_ + len > capacity_) {
      memcpy(dst, src, capacity_ - read_pos_);
      len -= capacity_ - read_pos_;
      dst += capacity_ - read_pos_;
      read_pos_ = 0;
      bool file_end = false;
      capacity_ = file_reader_->Read(static_cast<char*>(buffer_), BUFFER_SIZE,
                                     &file_end);
      src = static_cast<char*>(buffer_) + read_pos_;
      if (file_end) {
        break;
      }
    }
    if (read_pos_ + len > capacity_) {
      return false;
    }
    memcpy(dst, src, len);
    read_pos_ += len;
    return true;
  }

 private:
  std::shared_ptr<galileo::utils::IFileReader> file_reader_;
  size_t read_pos_;
  size_t capacity_;
  void* buffer_;
};
}  // namespace service
}  // namespace galileo
