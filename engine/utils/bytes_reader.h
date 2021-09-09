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

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>

namespace galileo {
namespace utils {

class BytesReader {
 public:
  explicit BytesReader(const char* raw_bytes)
      : BytesReader(raw_bytes, UINT32_MAX) {}

  BytesReader(const char* raw_bytes, size_t size) {
    raw_bytes_ = raw_bytes;
    total_size_ = size;
    start_pos_ = 0;
  }

  template <typename T>
  bool Read(T* result);

  inline bool Read(char* result, size_t len);

  inline bool Peek(uint16_t* result);

 private:
  size_t total_size_;
  size_t start_pos_;
  const char* raw_bytes_;
};

template <typename T>
bool BytesReader::Read(T* result) {
  if (total_size_ < start_pos_ + sizeof(T)) {
    return false;
  }

  *result = *reinterpret_cast<const T*>(raw_bytes_ + start_pos_);
  start_pos_ += sizeof(T);
  return true;
}
bool BytesReader::Read(char* result, size_t length) {
  if (total_size_ < start_pos_ + length) {
    return false;
  }

  memcpy(result, raw_bytes_ + start_pos_, length);
  start_pos_ += length;
  return true;
}

bool BytesReader::Peek(uint16_t* result) {
  if (total_size_ < start_pos_ + 2) {
    return false;
  }

  *result = *reinterpret_cast<const uint16_t*>(raw_bytes_ + start_pos_);
  return true;
}

}  // namespace utils
}  // namespace galileo
