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

#include <mutex>

namespace galileo {
namespace utils {

class ListChunk {
 public:
  ListChunk(size_t record_count, size_t record_size, ListChunk* pre);
  ~ListChunk();
  char* MallocRecord(size_t record_size);
  size_t RecordCount() { return record_count_; }
  ListChunk* Next() { return next_; }

 private:
  size_t record_count_;  // records numbers of current chunk
  char* head_;           // head record of current chunk
  char* cur_;
  ListChunk* next_;  // next free chunk
};

class MemoryPool {
 public:
  MemoryPool(size_t record_count, size_t record_size)
      : record_count_(record_count),
        record_size_(record_size),
        head_chunk_(nullptr),
        cur_chunk_(nullptr) {}

  ~MemoryPool() {
    ListChunk* next = head_chunk_;
    while (next != nullptr) {
      head_chunk_ = next;
      next = head_chunk_->Next();
      delete head_chunk_;
    }
    head_chunk_ = nullptr;
    cur_chunk_ = nullptr;
  }

  char* NextRecord();

 private:
  void _AllocateNextChunk();

 private:
  // blockSize
  size_t record_count_;
  // record size(one vertex or edge)
  size_t record_size_;
  // free records list
  ListChunk* head_chunk_;
  ListChunk* cur_chunk_;
  // lock
  std::mutex mutex_;
};

}  // namespace utils
}  // namespace galileo
