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

#include <cstring>
#include <initializer_list>

#include <assert.h>
#include <stdlib.h>

#include "common/message.h"
#include "common/types.h"
#include "glog/logging.h"

namespace galileo {
namespace common {

class Packer {
 public:
  Packer(std::string *str) : buffer_(str), offset_(0), resize_num_(0) {}
  Packer(std::string *str, size_t initial_size)
      : buffer_(str), offset_(0), resize_num_(0) {
    if (str->size() < initial_size) {
      this->_Resize(initial_size);
    }
  }

 public:
  template <typename T>
  size_t Pack(const T &arg);

  template <typename T, typename... Types>
  size_t Pack(const T &first, Types... args);

  template <typename T>
  bool UnPack(T *arg);

  template <typename T, typename... Types>
  bool UnPack(T *first, Types... args);

  size_t PackEnd() {
    buffer_->resize(offset_);
    return buffer_->size();
  }

  template <typename T>
  bool PackWithOffset(size_t offset, T &data);

 private:
  template <typename T>
  size_t _PackImp(const std::vector<T> &obj);

  template <typename T>
  size_t _PackImp(const T &obj);

  template <typename T>
  size_t _PackImp(const ArraySpec<T> &obj);

  template <typename T>
  bool _UnPackImp(std::vector<T> *obj);

  template <typename T>
  bool _UnPackImp(T *obj);

  template <typename T>
  bool _UnPackImp(ArraySpec<T> *obj);

 private:
  size_t _Resize(size_t min_size) {
    do {
      ++resize_num_;
      if (galileo::common::MEMORY_SIZE * resize_num_ >= min_size) {
        break;
      }
    } while (1);

    buffer_->resize(galileo::common::MEMORY_SIZE * resize_num_);
    return buffer_->size();
  }

 private:
  std::string *buffer_;
  size_t offset_;
  size_t resize_num_;
};

template <typename T>
size_t Packer::Pack(const T &arg) {
  return this->_PackImp(arg);
}

template <typename T, typename... Types>
size_t Packer::Pack(const T &first, Types... args) {
  this->_PackImp(first);
  return this->Pack(args...);
}

template <typename T>
bool Packer::PackWithOffset(size_t offset, T &data) {
  size_t tmp_offset = offset_;
  offset_ = offset;
  this->_PackImp(data);
  offset_ = tmp_offset;
  return true;
}

template <typename T>
bool Packer::UnPack(T *arg) {
  return this->_UnPackImp(arg);
}

template <typename T, typename... Types>
bool Packer::UnPack(T *first, Types... args) {
  if (!this->_UnPackImp(first)) {
    return false;
  }
  return this->UnPack(args...);
}

template <typename T>
size_t Packer::_PackImp(const std::vector<T> &obj) {
  size_t start_offset = offset_;
  size_t obj_num = obj.size();
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(obj_num))) {
    this->_Resize(offset_ + sizeof(obj_num));
  }
  *(size_t *)(buffer_->c_str() + offset_) = obj_num;
  offset_ += sizeof(obj_num);
  for (auto &elem : obj) {
    this->_PackImp(elem);
  }
  return start_offset;
}

template <typename T>
size_t Packer::_PackImp(const T &obj) {
  size_t start_offset = offset_;
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(obj))) {
    this->_Resize(offset_ + sizeof(obj));
  }
  *(T *)(buffer_->c_str() + offset_) = obj;
  offset_ += sizeof(obj);
  return start_offset;
}

template <typename T>
size_t Packer::_PackImp(const ArraySpec<T> &obj) {
  size_t start_offset = offset_;
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < obj.Capacity())) {
    this->_Resize(offset_ + obj.Capacity());
  }
  *(size_t *)(buffer_->c_str() + offset_) = obj.cnt;
  offset_ += sizeof(obj.cnt);
  for (size_t i = 0; i < obj.cnt; ++i) {
    *(T *)(buffer_->c_str() + offset_) = *(obj.data + i);
    offset_ += sizeof(T);
  }
  return start_offset;
}

template <typename T>
bool Packer::_UnPackImp(std::vector<T> *obj) {
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(size_t))) {
    return false;
  }
  size_t obj_size = *(size_t *)(buffer_->c_str() + offset_);
  obj->resize(obj_size);
  offset_ += sizeof(obj_size);
  for (size_t i = 0; i < obj_size; ++i) {
    if (!this->_UnPackImp(&obj->at(i))) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Packer::_UnPackImp(T *obj) {
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(*obj))) {
    return false;
  }
  *obj = *(T *)(buffer_->c_str() + offset_);
  offset_ += sizeof(*obj);
  return true;
}

template <typename T>
bool Packer::_UnPackImp(ArraySpec<T> *obj) {
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(obj->cnt))) {
    return false;
  }
  obj->cnt = *(size_t *)(buffer_->c_str() + offset_);
  offset_ += sizeof(obj->cnt);
  if ((buffer_->size() < offset_) ||
      (buffer_->size() - offset_ < sizeof(T) * obj->cnt)) {
    return false;
  }
  obj->data = (T *)(buffer_->c_str() + offset_);
  offset_ += sizeof(T) * obj->cnt;
  return true;
}

}  // namespace common
}  // namespace galileo
