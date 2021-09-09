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

#include "utils/memory_pool.h"

namespace galileo {
namespace utils {

ListChunk::ListChunk(size_t record_count, size_t record_size, ListChunk* pre)
    : record_count_(record_count) {
  if (pre != nullptr) {
    pre->next_ = this;
  }
  this->next_ = nullptr;
  this->head_ = (char*)malloc(record_count * record_size);
  cur_ = head_;
}

ListChunk::~ListChunk() {
  if (head_ != nullptr) {
    delete head_;
  }
  cur_ = nullptr;
}
char* ListChunk::MallocRecord(size_t record_size) {
  if (0 == record_count_) {
    return nullptr;
  }
  char* memory = cur_;
  cur_ += record_size;
  --record_count_;
  return memory;
}

void MemoryPool::_AllocateNextChunk() {
  cur_chunk_ =
      new ListChunk(this->record_count_, this->record_size_, cur_chunk_);
  if (head_chunk_ == nullptr) {
    head_chunk_ = cur_chunk_;
  }
}

char* MemoryPool::NextRecord() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (this->cur_chunk_ == nullptr || this->cur_chunk_->RecordCount() <= 0) {
    _AllocateNextChunk();
  }
  return this->cur_chunk_->MallocRecord(this->record_size_);
}

}  // namespace utils
}  // namespace galileo
