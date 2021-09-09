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

#include "service/edge.h"

#include <string>
#include <vector>

#include "common/macro.h"
#include "common/schema.h"
#include "common/singleton.h"
#include "glog/logging.h"
#include "utils/bytes_reader.h"

namespace galileo {
namespace service {
using Schema = galileo::schema::Schema;

Edge::Edge(uint8_t etype) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  raw_data_ = galileo::common::Singleton<EntityPoolManager>::GetInstance()
                  ->GetEMemoryPool(etype)
                  ->NextRecord();
  memset(raw_data_, 0, schema->GetERecordSize(etype));
}
Edge::~Edge() {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t etype = GetType();
  for (size_t i = 0; i < schema->GetEFieldCount(etype); ++i) {
    if (schema->IsEVarField(etype, i)) {
      std::string attr = schema->GetEFieldName(etype, i);
      int var_index = schema->GetEVarFieldIdx(etype, attr);
      assert(var_index >= 0);
      if (!_IsDirectStore(var_index)) {
        int attr_idx = schema->GetEFieldOffset(etype, attr);
        assert(attr_idx >= 0);
        uint64_t var_address =
            *reinterpret_cast<const uint64_t*>(raw_data_ + attr_idx);
        char* real_data = (char*)var_address;
        if (real_data) free(real_data);
      }
    }
  }
}

galileo::common::VertexID Edge::GetSrcVertex() const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  int offset = schema->GetEFieldOffset(GetType(), SCM_ENTITY_1);
  assert(offset >= 0);
  return *reinterpret_cast<const galileo::common::VertexID*>(raw_data_ +
                                                             offset);
}

galileo::common::VertexID Edge::GetDstVertex() const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  int offset = schema->GetEFieldOffset(GetType(), SCM_ENTITY_2);
  assert(offset >= 0);
  return *reinterpret_cast<const galileo::common::VertexID*>(raw_data_ +
                                                             offset);
}

float Edge::GetWeight() const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  int offset = schema->GetEFieldOffset(GetType(), SCM_WEIGHT);
  if (offset < 0) {
    return 1.0;
  } else {
    return *reinterpret_cast<const float*>(raw_data_ + offset);
  }
}

float Edge::GetFixFeatureWithOffset(size_t offset,
                                    const std::string& attr_type) const {
  float result = 0;
  if (attr_type == "DT_UINT8") {
    result = static_cast<float>(
        *reinterpret_cast<const uint8_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_UINT16") {
    result = static_cast<float>(
        *reinterpret_cast<const uint16_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_UINT32") {
    result = static_cast<float>(
        *reinterpret_cast<const uint32_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_UINT64") {
    result = static_cast<float>(
        *reinterpret_cast<const uint64_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_INT8") {
    result = static_cast<float>(
        *reinterpret_cast<const int8_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_INT16") {
    result = static_cast<float>(
        *reinterpret_cast<const int16_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_INT32") {
    result = static_cast<float>(
        *reinterpret_cast<const int32_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_INT64") {
    result = static_cast<float>(
        *reinterpret_cast<const int64_t*>(raw_data_ + offset));
  } else if (attr_type == "DT_FLOAT") {
    result =
        static_cast<float>(*reinterpret_cast<const float*>(raw_data_ + offset));
  } else if (attr_type == "DT_DOUBLE") {
    result = static_cast<float>(
        *reinterpret_cast<const double*>(raw_data_ + offset));
  } else {
    result = 0;
  }
  return result;
}

const char* Edge::GetFeature(const std::string& attr_name) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t etype = GetType();
  int tmp_fid = schema->GetEFieldIdx(etype, attr_name);
  if (tmp_fid < 0) {
    return nullptr;
  }
  size_t fid = static_cast<size_t>(tmp_fid);
  int begin_idx = schema->GetEFieldOffset(etype, attr_name);
  if (begin_idx < 0) {
    return nullptr;
  }
  if (schema->IsEVarField(etype, fid) && !_IsDirectStore(attr_name)) {
    uint64_t var_address =
        *reinterpret_cast<const uint64_t*>(raw_data_ + begin_idx);
    return reinterpret_cast<const char*>(var_address);
  } else {
    return reinterpret_cast<const char*>(raw_data_ + begin_idx);
  }
}

bool Edge::DeSerialize(uint8_t type, const char* s, size_t size) {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  galileo::utils::BytesReader bytes_reader(s, size);
  size_t write_off = schema->GetEFixedFieldLen(type);
  if (write_off > size) {
    LOG(ERROR) << " Fix attribute info error."
               << " write_off:" << write_off << " ,size:" << size;
    return false;
  }
  // fix attr
  if (!bytes_reader.Read(raw_data_, write_off)) {
    LOG(ERROR) << " Fix attribute info error";
    return false;
  }

  // var attr
  uint16_t var_len;
  for (size_t i = 0; i < schema->GetEVarFieldCount(type); i++) {
    if (!bytes_reader.Peek(&var_len)) {
      LOG(ERROR) << " Var attribute length error";
      return false;
    }
    if (_IsDirectStore(i)) {
      if (!bytes_reader.Read(raw_data_ + write_off, var_len + 2)) {
        LOG(ERROR) << " Var attribute info error";
        return false;
      }
    } else {
      char* var_attr = (char*)malloc(var_len + 2);
      if (!bytes_reader.Read(var_attr, var_len + 2)) {
        LOG(ERROR) << " Var attribute info error";
        return false;
      }
      uint64_t var_address = (uint64_t)var_attr;
      memcpy(raw_data_ + write_off, &var_address, 8);
    }
    write_off += 8;
  }
  return true;
}

bool Edge::_IsDirectStore(const std::string& field_name) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  uint8_t etype = GetType();
  int var_index = schema->GetEVarFieldIdx(etype, field_name);
  if (var_index < 0) {
    LOG(ERROR) << " Get edge var field idx fail."
               << " edge type:" << etype << " ,field name:" << field_name;
    return false;
  }
  return _IsDirectStore(static_cast<size_t>(var_index));
}

bool Edge::_IsDirectStore(size_t var_index) const {
  Schema* schema = galileo::common::Singleton<Schema>::GetInstance();
  char state =
      *(raw_data_ + schema->GetEStateOffset(GetType()) + (var_index / 8));
  return state & (1 << (7 - (var_index % 8)));
}

}  // namespace service
}  // namespace galileo
