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

#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace galileo {
namespace utils {

class Buffer {
 protected:
  char* m_buffer;
  size_t m_capacity;
  size_t m_size;
  size_t m_growStep;

 public:
  Buffer(size_t capacity = 0, size_t growStep = 0)
      : m_buffer(NULL), m_capacity(0), m_size(0), m_growStep(growStep) {
    this->reserve(capacity);
  }
  ~Buffer() { this->destroy(); }

 public:
  void reserve(size_t capacity, size_t growStep = 0) {
    if (capacity <= m_capacity) return;
    if (growStep) {
      m_growStep = this->normalizeCapacity(growStep);
    }
    capacity = this->calcCapacity(capacity);
    if (!m_buffer) {
      m_buffer = (char*)malloc(capacity);
      m_capacity = capacity;
    } else {
      char* buf = (char*)malloc(capacity);
      if (m_size > 0) memcpy(buf, m_buffer, m_size);
      free(m_buffer);
      m_buffer = buf;
      m_capacity = capacity;
    }
  }

  char* buffer() { return m_buffer; }

  size_t capacity() const { return m_capacity; }

  size_t size() const { return m_size; }

  size_t avail() const { return m_capacity - m_size; }

  bool empty() const { return m_size < 1; }

  void clear() { m_size = 0; }

  const char* readBuffer(size_t* length = NULL) {
    if (!m_buffer) return NULL;
    if (length) *length = m_size;
    return m_buffer;
  }

  char* writeBuffer(size_t* length = NULL) {
    if (!m_buffer) return NULL;
    if (length) *length = m_capacity - m_size;
    return m_buffer + m_size;
  }

  bool write(const void* data, size_t length) {
    this->reserve(m_size + length);
    char* dst = m_buffer + m_size;
    memcpy(dst, data, length);
    m_size += length;
    return true;
  }
  void cut(size_t length) {
    assert(length <= m_size);
    if (length < m_size) {
      char* src = m_buffer + length;
      char* dst = m_buffer;
      memmove(dst, src, m_size - length);
      m_size -= length;
    } else {
      m_size = 0;
    }
  }

  size_t advance(size_t length) {
    assert(m_size + length <= m_capacity);
    m_size += length;
    return m_size;
  }

 protected:
  size_t normalizeCapacity(size_t capacity) {
    size_t i = 0;
    size_t nc = 0;
    while (i < 32 && nc < capacity) {
      nc = (1 << i);
      ++i;
    }
    return nc;
  }
  size_t calcCapacity(size_t want) {
    if (!m_growStep) return (want + 4095) & (~4095);
    size_t capacity = m_capacity;
    while (capacity < want) capacity += m_growStep;
    return capacity;
  }
  void destroy() {
    if (m_buffer) {
      free(m_buffer);
      m_capacity = 0;
      m_size = 0;
    }
  }
};

}  // namespace utils
}  // namespace galileo
