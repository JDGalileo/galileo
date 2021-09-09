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

#include <assert.h>
#include <memory>

#include "common/macro.h"
#include "common/schema.h"
#include "common/schema_checker.h"

#include "glog/logging.h"

namespace galileo {
namespace schema {

Schema::~Schema() {
  if (vfields_idx_ != NULL) {
    delete[] vfields_idx_;
  }

  if (vfields_dtype_ != NULL) {
    delete[] vfields_dtype_;
  }

  if (efields_idx_ != NULL) {
    delete[] efields_idx_;
  }

  if (efields_dtype_ != NULL) {
    delete[] efields_dtype_;
  }
}

bool Schema::Build(const char* content) {
  std::string errMsg;
  quickjson::Value schema;
  if (unlikely(!schema.parse(content, &errMsg))) {
    LOG(ERROR) << "Parse schema file fail!"
               << "errno:" << errno << ",errMsg:" << errMsg;
    return false;
  }

  SchemaChecker checker;

  if (unlikely(!checker.CheckFirstLevelValidity(schema))) {
    LOG(ERROR) << " The first level format of schema is invalid.";
    return false;
  }

  const quickjson::Value& vertexes = schema["vertexes"];
  if (unlikely(!checker.CheckVertexOrEdgeValidity(vertexes, "vertex"))) {
    LOG(ERROR) << " The vertex format of schema is invalid.";
    return false;
  }

  const quickjson::Value& edges = schema["edges"];
  if (unlikely(!checker.CheckVertexOrEdgeValidity(edges, "edge"))) {
    LOG(ERROR) << " The edge format of schema is invalid.";
    return false;
  }

  if (unlikely(!this->_BuildVertexesSchema(vertexes))) {
    LOG(ERROR) << " Build vertex schema fail.";
    return false;
  }

  if (unlikely(!this->_BuildEdgesSchema(edges))) {
    LOG(ERROR) << " Build edge schema fail.";
    return false;
  }
  return true;
}

std::string Schema::GetVFieldDtype(uint8_t vtype, size_t idx) noexcept {
  assert(vtype < vtype_num_);
  auto& fields = vfields_dtype_[vtype];
  assert(idx < fields.size());
  return fields[idx];
}

int Schema::GetVFieldIdx(uint8_t vtype,
                         const std::string& field_name) noexcept {
  assert(vtype < vtype_num_);
  auto& fields_idx = vfields_idx_[vtype];
  auto itor = fields_idx.find(field_name);
  if (itor == fields_idx.end()) {
    LOG(ERROR) << " Can not find field name."
               << " vertex type:" << std::to_string(vtype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

size_t Schema::GetVFieldCount(uint8_t vtype) noexcept {
  assert(vtype < vtype_num_);
  return vfields_idx_[vtype].size();
}

size_t Schema::GetVFixedFieldsLen(uint8_t vtype) noexcept {
  assert(vtype < vtype_num_);
  return vfixed_field_len_[vtype];
}

int Schema::GetVFieldOffset(uint8_t vtype,
                            const std::string& field_name) noexcept {
  assert(vtype < vtype_num_);
  auto& fields_offset = vfields_offset_[vtype];
  auto itor = fields_offset.find(field_name);
  if (itor == fields_offset.end()) {
    LOG(ERROR) << " Can not find field name in vfields_offset."
               << " vertex type:" << std::to_string(vtype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

bool Schema::IsVVarField(uint8_t vtype, size_t idx) noexcept {
  assert(vtype < vtype_num_);
  std::string dtype = GetVFieldDtype(vtype, idx);
  if (std::string::npos != dtype.find("ARRAY") ||
      std::string::npos != dtype.find("STRING") ||
      std::string::npos != dtype.find("BINARY")) {
    return true;
  }
  return false;
}

int Schema::GetVVarFieldIdx(uint8_t vtype,
                            const std::string& field_name) noexcept {
  assert(vtype < vtype_num_);
  auto& var_fields_idx = vfields_var_idx_[vtype];
  auto itor = var_fields_idx.find(field_name);
  if (itor == var_fields_idx.end()) {
    LOG(ERROR) << "Can not find field name."
               << " vertex type:" << std::to_string(vtype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

size_t Schema::GetVRecordSize(uint8_t vtype) noexcept {
  assert(vtype < vtype_num_);
  size_t record_size = 0;
  FieldIdx::const_iterator it = vfields_idx_[vtype].begin();
  for (; it != vfields_idx_[vtype].end(); ++it) {
    size_t idx = it->second;
    std::string dtype = GetVFieldDtype(vtype, idx);
    record_size += _GetFieldLenByDType(dtype);
  }
  size_t var_field_size = GetVVarFieldCount(vtype);
  size_t state_size = var_field_size / 8;
  if (var_field_size % 8 > 0) {
    ++state_size;
  }
  record_size += state_size;
  return record_size;
}

int Schema::GetVStateOffset(uint8_t vtype) noexcept {
  assert(vtype < vtype_num_);
  return GetVFieldOffset(vtype, "state");
}

std::string Schema::GetEFieldDtype(uint8_t etype, size_t idx) noexcept {
  assert(etype < etype_num_);
  auto& fields = efields_dtype_[etype];
  assert(idx < fields.size());
  return fields[idx];
}

int Schema::GetEFieldIdx(uint8_t etype,
                         const std::string& field_name) noexcept {
  assert(etype < etype_num_);
  auto& fields_idx = efields_idx_[etype];
  auto itor = fields_idx.find(field_name);
  if (itor == fields_idx.end()) {
    LOG(ERROR) << " Can not find field name."
               << " edge type:" << std::to_string(etype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

size_t Schema::GetEFieldCount(uint8_t etype) noexcept {
  assert(etype < etype_num_);
  return efields_idx_[etype].size();
}

size_t Schema::GetEFixedFieldLen(uint8_t etype) noexcept {
  assert(etype < etype_num_);
  return efixed_field_len_[etype];
}

int Schema::GetEFieldOffset(uint8_t etype,
                            const std::string& field_name) noexcept {
  assert(etype < etype_num_);
  auto& fields_offset = efields_offset_[etype];
  auto itor = fields_offset.find(field_name);
  if (itor == fields_offset.end()) {
    LOG(ERROR) << " Can not find field name in vfields_offset."
               << " edge type:" << std::to_string(etype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

bool Schema::IsEVarField(uint8_t etype, size_t idx) noexcept {
  assert(etype < etype_num_);
  std::string dtype = GetEFieldDtype(etype, idx);
  if (std::string::npos != dtype.find("ARRAY") ||
      std::string::npos != dtype.find("STRING") ||
      std::string::npos != dtype.find("BINARY")) {
    return true;
  }
  return false;
}

int Schema::GetEVarFieldIdx(uint8_t etype,
                            const std::string& field_name) noexcept {
  assert(etype < etype_num_);
  auto& var_fields_idx = efields_var_idx_[etype];
  auto itor = var_fields_idx.find(field_name);
  if (itor == var_fields_idx.end()) {
    LOG(ERROR) << "Can not find field name."
               << " edge type:" << std::to_string(etype)
               << " ,filed_name:" << field_name;
    return -1;
  }
  return static_cast<int>(itor->second);
}

size_t Schema::GetERecordSize(uint8_t etype) noexcept {
  assert(etype < etype_num_);
  size_t record_size = 0;
  FieldIdx::const_iterator it = efields_idx_[etype].begin();
  for (; it != efields_idx_[etype].end(); ++it) {
    size_t idx = static_cast<size_t>(it->second);
    const std::string& dtype = GetEFieldDtype(etype, idx);
    record_size += _GetFieldLenByDType(dtype);
  }
  size_t var_field_count = GetEVarFieldCount(etype);
  size_t state_size = var_field_count / 8;
  if (var_field_count % 8 > 0) {
    ++state_size;
  }
  record_size += state_size;
  return record_size;
}
int Schema::GetEStateOffset(uint8_t etype) noexcept {
  assert(etype < etype_num_);
  return GetEFieldOffset(etype, "state");
}

bool Schema::_BuildVertexesSchema(const quickjson::Value& vertexes) {
  std::string type_name(SCM_VTYPE);
  if (!_LoadFieldSchema(vfields_idx_, vfields_dtype_, vfields_name_, vertexes,
                        type_name, vtype_num_)) {
    return false;
  }
  if (!_LoadFieldsVarIdx(vertexes, type_name, vtype_num_, vfields_var_idx_)) {
    return false;
  }
  if (!_LoadFixedFieldLen(vertexes, type_name, vfields_var_idx_,
                          vfixed_field_len_)) {
    return false;
  }
  return _BuildOffset(vertexes, type_name, vtype_num_, vfields_var_idx_,
                      vfields_offset_);
}

bool Schema::_BuildEdgesSchema(const quickjson::Value& edges) {
  std::string type_name(SCM_ETYPE);
  if (!_LoadFieldSchema(efields_idx_, efields_dtype_, efields_name_, edges,
                        type_name, etype_num_)) {
    return false;
  }

  if (!_LoadFieldsVarIdx(edges, type_name, etype_num_, efields_var_idx_)) {
    return false;
  }

  if (!_LoadFixedFieldLen(edges, type_name, efields_var_idx_,
                          efixed_field_len_)) {
    return false;
  }
  return _BuildOffset(edges, type_name, etype_num_, efields_var_idx_,
                      efields_offset_);
}

bool Schema::_BuildOffset(const quickjson::Value& values,
                          const std::string& type_name, uint8_t type_num,
                          FieldIdx* fields_var_idx,
                          FieldOffset*& field_offset) {
  if (unlikely(!values.isArray())) {
    LOG(ERROR) << " The vertexes schema is not array!";
    return false;
  }
  std::string field_name, field_dtype;
  if (likely(NULL == field_offset)) {
    field_offset = new FieldOffset[type_num];
  }

  std::unique_ptr<size_t> begin_offset(new size_t[type_num]());
  if (!_LoadOffset(values, type_name, begin_offset.get(), true, field_offset)) {
    return false;
  }

  for (size_t i = 0; i < static_cast<size_t>(type_num); ++i) {
    size_t var_size = fields_var_idx[i].size();
    size_t state_byte_size = var_size / 8;
    if (var_size % 8 > 0) {
      ++state_byte_size;
    }
    field_offset[i].emplace("state", begin_offset.get()[i]);
    begin_offset.get()[i] += state_byte_size;
  }

  if (!_LoadOffset(values, type_name, begin_offset.get(), false,
                   field_offset)) {
    return false;
  }
  return true;
}

size_t Schema::_GetFieldLenByDType(const std::string& dtype,
                                   bool except_array) {
  if ((!except_array && dtype.find("ARRAY") != std::string::npos) ||
      dtype.find("STRING") != std::string::npos ||
      dtype.find("BINARY") != std::string::npos) {
    return sizeof(int64_t);
  } else if (dtype.find("UINT8") != std::string::npos) {
    return sizeof(uint8_t);
  } else if (dtype.find("INT8") != std::string::npos) {
    return sizeof(int8_t);
  } else if (dtype.find("UINT16") != std::string::npos) {
    return sizeof(uint16_t);
  } else if (dtype.find("INT16") != std::string::npos) {
    return sizeof(int16_t);
  } else if (dtype.find("UINT32") != std::string::npos) {
    return sizeof(uint32_t);
  } else if (dtype.find("INT32") != std::string::npos) {
    return sizeof(int32_t);
  } else if (dtype.find("UINT64") != std::string::npos) {
    return sizeof(uint64_t);
  } else if (dtype.find("INT64") != std::string::npos) {
    return sizeof(int64_t);
  } else if (dtype.find("FLOAT") != std::string::npos) {
    return sizeof(float);
  } else if (dtype.find("DOUBLE") != std::string::npos) {
    return sizeof(double);
  } else if (dtype.find("BOOL") != std::string::npos) {
    return sizeof(uint8_t);
  } else {
    LOG(ERROR) << "No type : " << dtype;
  }
  return 0;
}

bool Schema::_LoadFieldSchema(FieldIdx*& field_idx, FieldDtype*& field_dtype,
                              std::vector<std::string>*& fields_name,
                              const quickjson::Value& values,
                              const std::string& type_name, uint8_t& type_num) {
  if (unlikely(!values.isArray())) {
    LOG(ERROR) << " The vertexes schema is not array!";
    return false;
  }
  int type = -1;
  std::string field_name, field_type;
  type_num = static_cast<uint8_t>(values.size());
  if (likely(NULL == field_idx)) {
    field_idx = new FieldIdx[type_num];
    field_dtype = new FieldDtype[type_num];
    fields_name = new std::vector<std::string>[type_num];
  }
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    quickjson::Object cur_value = values[idx].toObject();
    type = cur_value[type_name];
    assert(type == idx);
    for (int pos = 0; pos < static_cast<int>(cur_value.size()); ++pos) {
      field_name = cur_value.getName(pos).getString(NULL);
      if (unlikely(field_name == SCM_ATTRS)) {
        auto fun_attrs = [&](quickjson::Object& obj, size_t attr_idx) {
          field_idx[type].emplace(obj[SCM_FIELD_NAME].getString(NULL),
                                  pos + attr_idx);
          field_dtype[type].emplace_back(obj[SCM_FIELD_TYPE].getString(NULL));
          fields_name[type].emplace_back(obj[SCM_FIELD_NAME].getString(NULL));
          return true;
        };

        if (!_TraversalFieldAttrs(cur_value.at(pos), fun_attrs)) {
          return false;
        }
        continue;
      }
      field_idx[type].emplace(field_name, pos);
      if (field_name == SCM_VTYPE || field_name == SCM_ETYPE) {
        field_type = "DT_UINT8";
      } else {
        field_type = cur_value.at(pos).getString(NULL);
      }
      field_dtype[type].emplace_back(field_type);
      fields_name[type].emplace_back(field_name);
    }
  }
  return true;
}

bool Schema::_LoadOffset(const quickjson::Value& values,
                         const std::string& type_name, size_t* begin_offset,
                         bool is_fixed, FieldOffset*& field_offset) {
  std::string field_name, field_dtype;
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    quickjson::Object cur_value = values[idx].toObject();
    int tmp_type = cur_value[type_name];
    assert(tmp_type == idx);
    uint8_t type = static_cast<uint8_t>(tmp_type);
    for (int pos = 0; pos < static_cast<int>(cur_value.size()); ++pos) {
      field_name = cur_value.getName(pos).getString(NULL);
      if (unlikely(field_name == SCM_ATTRS)) {
        auto fun_attrs = [&](quickjson::Object& obj, size_t) {
          const char* _field_name = obj[SCM_FIELD_NAME].getString(NULL);
          const char* dtype = obj[SCM_FIELD_TYPE].getString(NULL);
          if (!_FillOffset(_field_name, dtype, is_fixed, type,
                           begin_offset[idx], field_offset)) {
            return false;
          }
          return true;
        };

        if (!_TraversalFieldAttrs(cur_value.at(pos), fun_attrs)) {
          return false;
        }
        continue;
      }

      if (field_name == SCM_VTYPE || field_name == SCM_ETYPE) {
        field_dtype = "DT_UINT8";
        if (is_fixed) {
          field_offset[type].emplace(field_name, begin_offset[idx]);
          begin_offset[idx] += sizeof(uint8_t);
        }
      } else {
        field_dtype = cur_value.at(pos).getString(NULL);

        if (!_FillOffset(field_name, field_dtype, is_fixed, type,
                         begin_offset[idx], field_offset)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool Schema::_FillOffset(const std::string& field_name,
                         const std::string& dtype, bool is_fixed, uint8_t type,
                         size_t& begin_offset, FieldOffset*& field_offset) {
  if (std::string::npos != dtype.find("ARRAY") ||
      std::string::npos != dtype.find("STRING") ||
      std::string::npos != dtype.find("BINARY")) {
    if (is_fixed) {
      return true;
    } else {
      size_t type_size = _GetFieldLenByDType(dtype);
      if (0 == type_size) {
        return false;
      }
      field_offset[type].emplace(field_name, begin_offset);
      begin_offset += type_size;
    }
  } else {
    if (!is_fixed) {
      return true;
    }
    size_t type_size = _GetFieldLenByDType(dtype);
    if (0 == type_size) {
      return false;
    }
    field_offset[type].emplace(field_name, begin_offset);
    begin_offset += type_size;
  }
  return true;
}

bool Schema::_LoadFieldsVarIdx(const quickjson::Value& values,
                               const std::string& type_name, uint8_t type_num,
                               FieldIdx*& fields_var_idx) {
  if (unlikely(!values.isArray())) {
    LOG(ERROR) << " The vertexes schema is not array!";
    return false;
  }
  int type = -1;
  std::string field_name, field_dtype;
  if (likely(NULL == fields_var_idx)) {
    fields_var_idx = new FieldIdx[type_num];
  }
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    size_t base_idx = 0;
    quickjson::Object cur_value = values[idx].toObject();
    type = cur_value[type_name];
    assert(type == idx);
    for (int pos = 0; pos < static_cast<int>(cur_value.size()); ++pos) {
      field_name = cur_value.getName(pos).getString(NULL);
      if (unlikely(field_name == SCM_ATTRS)) {
        auto fun_attrs = [&](quickjson::Object& obj, size_t) {
          std::string _field_name = obj[SCM_FIELD_NAME].getString(NULL);
          std::string field_type = obj[SCM_FIELD_TYPE].getString(NULL);
          if (std::string::npos != field_type.find("ARRAY") ||
              std::string::npos != field_type.find("STRING") ||
              std::string::npos != field_type.find("BINARY")) {
            fields_var_idx[type].emplace(_field_name, base_idx++);
          }
          return true;
        };
        if (!_TraversalFieldAttrs(cur_value.at(pos), fun_attrs)) {
          return false;
        }
        continue;
      }
      if (std::string::npos != field_name.find(SCM_ENTITY)) {
        field_dtype = cur_value.at(pos).getString(NULL);
        if (std::string::npos != field_dtype.find("STRING")) {
          fields_var_idx[type].emplace(field_name, base_idx++);
        }
      }
    }
  }
  return true;
}

bool Schema::_LoadFixedFieldLen(const quickjson::Value& values,
                                const std::string& type_name,
                                FieldIdx* fields_var_idx,
                                FixFieldLen& fixed_field_len) {
  fixed_field_len.clear();
  if (unlikely(!values.isArray())) {
    LOG(ERROR) << " The vertexes schema is not array!";
    return false;
  }
  std::string field_name, field_dtype;
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    size_t fixed_len = 0;
    quickjson::Object cur_value = values[idx].toObject();
    int tmp_type = cur_value[type_name];
    assert(tmp_type == idx);
    uint8_t type = static_cast<uint8_t>(tmp_type);
    for (int pos = 0; pos < static_cast<int>(cur_value.size()); ++pos) {
      field_name = cur_value.getName(pos).getString(NULL);
      if (unlikely(field_name == SCM_ATTRS)) {
        auto fun_attrs = [&](quickjson::Object& obj, size_t) {
          const std::string dtype = obj[SCM_FIELD_TYPE].getString(NULL);
          if (dtype.find("ARRAY") == std::string::npos &&
              dtype.find("STRING") == std::string::npos &&
              dtype.find("BINARY") == std::string::npos) {
            size_t type_len = _GetFieldLenByDType(dtype);
            if (0 == type_len) {
              return false;
            }
            fixed_len += type_len;
          }

          return true;
        };

        if (!_TraversalFieldAttrs(cur_value.at(pos), fun_attrs)) {
          return false;
        }
        continue;
      }
      field_dtype = "";
      if (field_name == SCM_VTYPE || field_name == SCM_ETYPE) {
        field_dtype = "DT_UINT8";
      } else {
        field_dtype = cur_value.at(pos).getString(NULL);
      }

      if (field_dtype.find("ARRAY") != std::string::npos ||
          field_dtype.find("STRING") != std::string::npos ||
          field_dtype.find("BINARY") != std::string::npos) {
        continue;
      }
      size_t type_len = this->_GetFieldLenByDType(field_dtype);
      if (0 == type_len) {
        return false;
      }
      fixed_len += type_len;
    }
    size_t state_len = fields_var_idx[type].size() / 8;
    if (fields_var_idx[type].size() % 8 > 0) {
      state_len++;
    }
    fixed_len += state_len;
    fixed_field_len[type] = fixed_len;
  }
  return true;
}

bool Schema::_TraversalFieldAttrs(
    const quickjson::Value& attrs,
    std::function<bool(quickjson::Object&, size_t attr_idx)> callback) {
  if (unlikely(!attrs.isArray())) {
    LOG(ERROR) << " The attrs schema is not array!";
    return false;
  }
  for (int idx = 0; idx < static_cast<int>(attrs.size()); ++idx) {
    quickjson::Object cur_value = attrs[idx].toObject();
    if (!callback(cur_value, idx)) {
      return false;
    }
  }
  return true;
}

size_t Schema::GetVFieldLen(uint8_t vtype, const std::string& field_name,
                            bool except_array) noexcept {
  return _GetFieldLenByDType(
      GetVFieldDtype(vtype, GetVFieldIdx(vtype, field_name)), except_array);
}

size_t Schema::GetEFieldLen(uint8_t etype, const std::string& field_name,
                            bool except_array) noexcept {
  return _GetFieldLenByDType(
      GetEFieldDtype(etype, GetEFieldIdx(etype, field_name)), except_array);
}

std::string Schema::GetVFieldName(uint8_t vtype, size_t idx) noexcept {
  assert(vtype < vtype_num_);
  auto& fields_name = vfields_name_[vtype];
  assert(idx < fields_name.size());
  return fields_name[idx];
}

std::string Schema::GetEFieldName(uint8_t etype, size_t idx) noexcept {
  assert(etype < etype_num_);
  auto& fields_name = efields_name_[etype];
  assert(idx < fields_name.size());
  return fields_name[idx];
}

size_t Schema::GetVVarFieldCount(uint8_t vtype) noexcept {
  assert(vtype < vtype_num_);
  return vfields_var_idx_[vtype].size();
}

size_t Schema::GetEVarFieldCount(uint8_t etype) noexcept {
  assert(etype < etype_num_);
  return efields_var_idx_[etype].size();
}

}  // namespace schema
}  // namespace galileo
