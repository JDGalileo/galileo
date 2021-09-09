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
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>

namespace quickjson {

class Value;
class Array;
class Object;

class PooledAllocator {
 public:
  static const bool kNeedFree = false;
  enum { kDefaultChunkSize = 64 * 1024 };
  enum { kMaxFreeChunk = 1024 * 1024 / kDefaultChunkSize };
  struct Chunk {
    Chunk* next;
    size_t capacity;
    size_t size;
    ///
    inline size_t avail() const { return capacity - size; }
  };

 protected:
  Chunk* m_head;
  int m_chunkCount;
  size_t m_chunkSize;
  Chunk* m_freeHead;
  int m_freeChunkCount;

 public:
  PooledAllocator(size_t chunkSize = kDefaultChunkSize)
      : m_head(NULL),
        m_chunkCount(0),
        m_chunkSize(chunkSize),
        m_freeHead(NULL),
        m_freeChunkCount(0) {}
  ~PooledAllocator() { this->destroy(); }

 public:
  void* alloc(size_t size) {
    if (!m_head || m_head->avail() < size) {
      this->allocChunk();
    }
    char* ret = (char*)m_head + sizeof(Chunk) + m_head->size;
    m_head->size += size;
    return ret;
  }
  void free(void* /*p*/) {}
  void clear() {
    while (m_head) {
      Chunk* p = m_head;
      m_head = p->next;
      this->collectChunk(p);
      m_chunkCount -= 1;
      assert(m_chunkCount >= 0);
    }
  }
  void destroy() {
    while (m_freeHead) {
      Chunk* p = m_freeHead;
      m_freeHead = p->next;
      ::free(p);
    }
    while (m_head) {
      Chunk* p = m_head;
      m_head = p->next;
      ::free(p);
    }
    m_chunkCount = 0;
    m_freeChunkCount = 0;
  }

 public:
  inline void Clear() { this->clear(); }
  inline size_t Capacity() const { return m_chunkSize * m_chunkSize; }
  inline size_t Size() const {
    if (!m_head) return 0;
    return ((m_chunkCount - 1) * m_chunkSize) + sizeof(Chunk) + m_head->size;
  }
  inline void* Malloc(size_t size) { return this->alloc(size); }
  inline void* Realloc(void* originalPtr, size_t originalSize, size_t newSize) {
    void* p = this->alloc(newSize);
    memcpy(p, originalPtr, originalSize);
    return p;
  }
  inline static void Free(void* /*ptr*/) {}

 protected:
  Chunk* allocChunk() {
    Chunk* chunk;
    if (m_freeHead) {
      chunk = m_freeHead;
      chunk->size = 0;
      m_freeHead = m_freeHead->next;
      m_freeChunkCount -= 1;
      assert(m_freeChunkCount >= 0);
      chunk->next = m_head;
      m_head = chunk;
      m_chunkCount += 1;
      return chunk;
    }
    chunk = (Chunk*)malloc(m_chunkSize);
    chunk->capacity = m_chunkSize - sizeof(Chunk);
    chunk->size = 0;
    if (m_head) {
      chunk->next = m_head;
      m_head = chunk;
    } else {
      chunk->next = NULL;
      m_head = chunk;
    }
    m_chunkCount += 1;
    return chunk;
  }
  void collectChunk(Chunk* p) {
    if (m_freeChunkCount >= kMaxFreeChunk) {
      ::free(p);
      return;
    }
    p->size = 0;
    p->next = m_freeHead;
    m_freeHead = p;
    m_freeChunkCount += 1;
  }
};

typedef rapidjson::UTF8<> encoding_type;
typedef rapidjson::GenericStringRef<char> StringRefType;
typedef PooledAllocator alloc_type;
typedef rapidjson::GenericValue<encoding_type, alloc_type> value_type;
typedef rapidjson::GenericStringBuffer<encoding_type> string_buffer;
typedef rapidjson::GenericMemberIterator<false, encoding_type,
                                         alloc_type>::Iterator miterator;
typedef rapidjson::GenericMemberIterator<true, encoding_type,
                                         alloc_type>::Iterator const_miterator;
typedef rapidjson::Type ValueType;

class OutputStringStream : public string_buffer {
 public:
  OutputStringStream(size_t initialCapacity = 64 * 1024)
      : string_buffer(NULL, initialCapacity) {}
  // void reserve(size_t capacity)
  //{
  //    capacity = (capacity + 4095) & (~4095);
  //    string_buffer::Reserve(capacity);
  //}
  size_t size() const { return string_buffer::GetSize(); }
  const char* data() const { return string_buffer::GetString(); }
  void clear() { string_buffer::Clear(); }
};
// Json Array value
class Array {
  friend class Value;
  friend class Object;

 protected:
  value_type* m_jsonval;
  alloc_type* m_allocator;

 protected:
  Array();

 public:
  Array(Value& value);
  Array(const Array& o);
  Array& operator=(const Array& o);
  ~Array();

 public:
  bool empty() const;
  size_t size() const;
  void reserve(size_t capacity);

  Array& append(const Value& value);

  Array& append(bool value);
  Array& append(int32_t value);
  Array& append(uint32_t value);
  Array& append(int64_t value);
  Array& append(uint64_t value);
  Array& append(const char* value);
  Array& append(const std::string& value);
  Array& append(double value);
  Array& append(const Array& value);
  Array& append(const Object& value);

  Object appendObject();
  Array appendArray();

  Value at(int index);
  Value at(int index) const;
  Value operator[](int index);
  Value operator[](int index) const;

 protected:
  value_type& getElement(int index) const;
  Array& operator=(const Value& v);
};
/// Json Object Value
class Object {
  friend class Value;
  friend class Array;

 protected:
  value_type* m_jsonval;
  alloc_type* m_allocator;

  // protected:
 public:
  Object(){};

 public:
  Object(Value& value);
  Object(const Object& o);
  Object& operator=(const Object& o);
  ~Object();

 public:
  bool empty() const;
  size_t size() const;
  // void reserve(size_t capacity);

  bool has(const char* name) const;
  bool has(const std::string& name) const;

  Object& set(const char* name, bool value);
  Object& set(const char* name, int32_t value);
  Object& set(const char* name, uint32_t value);
  Object& set(const char* name, int64_t value);
  Object& set(const char* name, uint64_t value);
  Object& set(const char* name, double value);
  Object& set(const char* name, const char* value);
  Object& set(const char* name, const std::string& value);
  Object& set(const char* name, const Value& value);
  Object& set(const char* name, const Array& value);
  Object& set(const char* name, const Object& value);

  Array addArray(const char* name);
  Object addObject(const char* name);

  Value at(int index) const;
  Value getName(int index) const;
  Value operator[](const char* name);
  Value operator[](const std::string& name);

 protected:
  bool _findMember(StringRefType& name, value_type** keypp,
                   value_type** valuepp);
  Object& operator=(const Value& v);
  value_type& _set(StringRefType& name, value_type& value,
                   bool if_not_exist = false);
};

// rapidjson::GenericValue<UTF8<>,char> 的一个代理
class Value {
  friend class Array;
  friend class Object;

 protected:
  alloc_type m_alloc;
  alloc_type* m_allocator;
  value_type m_jsonValue;
  value_type* m_jsonval;

 private:
  Array m_arr;
  Object m_obj;

 public:
  static alloc_type* newAllocator(size_t blockSize = 64 * 1024) {
    return new alloc_type(blockSize);
  }
  static void destroyAllocator(alloc_type* alloc) { delete alloc; }

 public:
  Value();
  Value(alloc_type& alloc);
  Value(const Value& o);
  Value& operator=(const Value& o);
  ~Value();

 public:
  bool parse(const char* jsonStr, std::string* errMsg = NULL);
  bool parseFile(const char* filepath, std::string* errMsg = NULL);

 public:
  Value& setNull();
  Array& toArray();
  Object& toObject();

  void clear();  // toNull

  Value& operator=(bool value);
  Value& operator=(int32_t value);
  Value& operator=(uint32_t value);
  Value& operator=(int64_t value);
  Value& operator=(uint64_t value);
  Value& operator=(double value);
  Value& operator=(const char* value);
  Value& operator=(const std::string& value);
  Value& operator=(const Array& value);
  Value& operator=(const Object& value);

  operator bool() const;
  operator int16_t() const;
  operator uint16_t() const;
  operator int32_t() const;
  operator uint32_t() const;
  operator int64_t() const;
  operator uint64_t() const;
  operator double() const;
  operator const char*() const;
  const char* getString(size_t* length) const;

  bool isNull() const;
  bool isBool() const;
  bool isInt() const;
  bool isUInt() const;
  bool isIntegral() const;
  bool isFloat() const;
  bool isDouble() const;
  bool isNumeric() const;
  bool isString() const;
  bool isArray() const;
  bool isObject() const;

  Array& asArray();
  Object& asObject();

  std::string asString() const;
  std::string asPrettyString() const;
  const char* serialize(OutputStringStream& os, size_t* length) const;

  template <typename RapidJsonStream>
  void serializeTo(RapidJsonStream& os) const {
    rapidjson::Writer<RapidJsonStream> writer(os);
    jsonValue().Accept(writer);
  }

 public:
  size_t size() const;
  Value operator[](int index);
  Value operator[](const char* key);
  Value operator[](int index) const;
  Value operator[](const char* key) const;

  template <typename T>
  Value& push_back(const T& v) {
    this->asArray().append(v);
    return *this;
  }
  template <typename T>
  Value& append(const T& v) {
    this->asArray().append(v);
    return *this;
  }

 public:
  Value(value_type& jsonVal, alloc_type& alloc)
      : m_allocator(&alloc), m_jsonval(&jsonVal), m_arr(*this), m_obj(*this) {}
  alloc_type& allocator() { return *m_allocator; }
  alloc_type& allocator() const { return *(alloc_type*)m_allocator; }
  value_type& jsonValue() { return *m_jsonval; }
  value_type& jsonValue() const { return *(value_type*)m_jsonval; }
};

}  // namespace quickjson
