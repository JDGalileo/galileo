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

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/singleton.h"
#include "quickjson/value.h"

namespace galileo {
namespace schema {

class Schema {
 public:
  using FieldDtype = std::vector<std::string>;
  using FieldIdx = std::unordered_map<std::string, size_t>;
  using FieldOffset = std::unordered_map<std::string, size_t>;
  using FixFieldLen = std::unordered_map<uint8_t, size_t>;
  using OrderMap = std::unordered_map<uint8_t, std::tuple<std::string, bool>>;

 public:
  ~Schema();

 public:
  bool Build(const char* content);

  std::string GetVFieldDtype(uint8_t vtype, size_t idx) noexcept;
  int GetVFieldIdx(uint8_t vtype, const std::string& field_name) noexcept;
  size_t GetVFieldCount(uint8_t vtype) noexcept;
  size_t GetVFixedFieldsLen(uint8_t vtype) noexcept;
  int GetVFieldOffset(uint8_t vtype, const std::string& field_name) noexcept;
  bool IsVVarField(uint8_t vtype, size_t idx) noexcept;
  int GetVVarFieldIdx(uint8_t vtype, const std::string& field_name) noexcept;
  size_t GetVRecordSize(uint8_t vtype) noexcept;
  int GetVStateOffset(uint8_t vtype) noexcept;
  uint8_t GetVTypeNum() noexcept { return vtype_num_; }
  size_t GetVFieldLen(uint8_t vtype, const std::string& field_name,
                      bool except_array = false) noexcept;
  std::string GetVFieldName(uint8_t vtype, size_t idx) noexcept;
  size_t GetVVarFieldCount(uint8_t vtype) noexcept;

  std::string GetEFieldDtype(uint8_t etype, size_t idx) noexcept;
  int GetEFieldIdx(uint8_t etype, const std::string& field_name) noexcept;
  size_t GetEFieldCount(uint8_t etype) noexcept;
  size_t GetEFixedFieldLen(uint8_t etype) noexcept;
  int GetEFieldOffset(uint8_t etype, const std::string& field_name) noexcept;
  bool IsEVarField(uint8_t etype, size_t idx) noexcept;
  int GetEVarFieldIdx(uint8_t etype, const std::string& field_name) noexcept;
  size_t GetERecordSize(uint8_t etype) noexcept;
  int GetEStateOffset(uint8_t etype) noexcept;
  uint8_t GetETypeNum() noexcept { return etype_num_; }
  size_t GetEFieldLen(uint8_t etype, const std::string& field_name,
                      bool except_array = false) noexcept;
  std::string GetEFieldName(uint8_t etype, size_t idx) noexcept;
  size_t GetEVarFieldCount(uint8_t etype) noexcept;

 private:
  bool _BuildVertexesSchema(const quickjson::Value& vertexes);
  bool _BuildEdgesSchema(const quickjson::Value& edges);
  bool _BuildOffset(const quickjson::Value& values,
                    const std::string& type_name, uint8_t type_num,
                    FieldIdx* fields_var_idx, FieldOffset*& field_offset);
  bool _LoadFieldSchema(FieldIdx*& field_idx, FieldDtype*& field_dtype,
                        std::vector<std::string>*& fields_name,
                        const quickjson::Value& values,
                        const std::string& type_name, uint8_t& type_num);
  bool _LoadOffset(const quickjson::Value& values, const std::string& type_name,
                   size_t* begin_offset, bool is_fixed,
                   FieldOffset*& field_offset);
  bool _FillOffset(const std::string& field_name, const std::string& dtype,
                   bool is_fixed, uint8_t type, size_t& begin_offset,
                   FieldOffset*& field_offset);
  bool _LoadFieldsVarIdx(const quickjson::Value& attrs,
                         const std::string& type_name, uint8_t type_num,
                         FieldIdx*& fields_var_idx);
  bool _LoadFixedFieldLen(const quickjson::Value& values,
                          const std::string& type_name,
                          FieldIdx* fields_var_idx,
                          FixFieldLen& fixed_field_len);

  bool _TraversalFieldAttrs(
      const quickjson::Value& attrs,
      std::function<bool(quickjson::Object&, size_t attr_idx)> callback);
  size_t _GetFieldLenByDType(const std::string& dtype,
                             bool except_array = false);

 private:
  // version schema
  // int version_ = -1;
  uint8_t vtype_num_ = 0;
  uint8_t etype_num_ = 0;

  // vertexes schema
  FieldIdx* vfields_idx_ = nullptr;
  FieldDtype* vfields_dtype_ = nullptr;
  FieldOffset* vfields_offset_ = nullptr;
  FieldIdx* vfields_var_idx_ = nullptr;
  std::vector<std::string>* vfields_name_ = nullptr;

  // edges schema
  FieldIdx* efields_idx_ = nullptr;
  FieldDtype* efields_dtype_ = nullptr;
  FieldOffset* efields_offset_ = nullptr;
  FieldIdx* efields_var_idx_ = nullptr;
  std::vector<std::string>* efields_name_ = nullptr;

  FixFieldLen efixed_field_len_;
  FixFieldLen vfixed_field_len_;
};

}  // namespace schema
}  // namespace galileo
