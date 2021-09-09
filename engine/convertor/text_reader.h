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

#include "utils/buffer.h"
#include "utils/file_reader.h"
#include "worker.h"

namespace galileo {
namespace convertor {

class TextReader {
 protected:
  galileo::utils::IFileReader* reader_;
  size_t read_pos_;
  size_t total_read_;
  bool feof_;
  Buffer buffer_;
  int64_t line_count_;

 public:
  TextReader(galileo::utils::IFileReader* reader,
             size_t buf_len = 16 * 1024 * 1024)
      : reader_(reader),
        read_pos_(0),
        total_read_(0),
        feof_(false),
        buffer_(buf_len),
        line_count_(0) {}
  ~TextReader() {}

 public:
  bool Load(Worker& worker) {
    size_t len = 0;
    while (1) {
      char* line = this->_NextLine(&len);
      if (!line) break;
      if (!worker(line, len)) return false;
      line_count_ += 1;
    }
    return true;
  }

 protected:
  size_t _ReadToBuffer() {
    buffer_.cut(read_pos_);
    read_pos_ = 0;
    size_t len = 0;
    char* buf = buffer_.writeBuffer(&len);
    if (0 >= len) {
      buffer_.reserve(buffer_.capacity() * 2);
      buf = buffer_.writeBuffer(&len);
    }

    size_t readed = reader_->Read(buf, len, &feof_);
    if (readed > 0) {
      buffer_.advance(readed);
      total_read_ += readed;
    }
    return readed;
  }

  char* _NextLine(size_t* len) {
    if (!total_read_) {
      this->_ReadToBuffer();
    }
    while (1) {
      char* rbuf = buffer_.buffer() + read_pos_;
      char* rend = buffer_.writeBuffer();

      char* line = rbuf;
      while (rbuf < rend) {
        if (*rbuf == '\n') {
          *rbuf = '\0';
          *len = (size_t)(rbuf - line);
          read_pos_ += *len + 1;
          return line;
        }
        rbuf += 1;
      }

      if (!feof_)
        this->_ReadToBuffer();
      else
        break;
    };

    if (read_pos_ < buffer_.size()) {
      buffer_.cut(read_pos_);
      read_pos_ = 0;
      buffer_.write("\n", 1);
      return this->_NextLine(len);
    }

    return nullptr;
  }
};

}  // namespace convertor
}  // namespace galileo
