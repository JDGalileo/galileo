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

#include "common/schema_checker.h"

#include <errno.h>

#include "common/macro.h"
#include "glog/logging.h"
#include "proto/types.pb.h"

namespace galileo {
namespace schema {

SchemaChecker::SchemaChecker() { this->_InitPredefinedFields(); }

SchemaChecker::~SchemaChecker() {
  main_fields_.clear();
  sub_fields_.clear();
}

bool SchemaChecker::CheckFirstLevelValidity(quickjson::Value& root) {
  quickjson::Object& obj = root.toObject();
  if (obj.size() != main_fields_.size()) {
    LOG(ERROR) << " The number of schema main fields is invalid!";
    return false;
  }

  for (size_t idx = 0; idx < main_fields_.size(); ++idx) {
    int pos = static_cast<int>(idx);
    const char* field_name = (const char*)(obj.getName(pos));
    if (unlikely(std::string::npos == main_fields_[pos].find(field_name))) {
      LOG(ERROR) << " The name of first level schema is invalid!"
                 << " expection:" << main_fields_[pos]
                 << " ,actuality:" << field_name;
      return false;
    }
  }

  return true;
}

bool SchemaChecker::CheckVertexOrEdgeValidity(const quickjson::Value& values,
                                              const std::string& main_field) {
  if (unlikely(!values.isArray())) {
    LOG(ERROR) << " The vertex or edge schema is not Array class!";
    return false;
  }

  const std::vector<std::string>& pre_fields =
      _GetPredefinedSubFieldsName(main_field);
  int pre_pos = 0, pos = 0;
  std::string sub_field, dtype, pre_sub_field;
  galileo::proto::DataType enum_dt;
  quickjson::Object obj;

  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    pos = pre_pos = 0;
    obj = values[idx].toObject();
    while (pos < static_cast<int>(obj.size())) {
      sub_field = (const char*)(obj.getName(pos));
      if (unlikely(pre_fields[pre_pos] != sub_field)) {
        LOG(ERROR) << " The name of vertexes or edges schema is invalid!"
                   << " ecpection:" << pre_fields[pre_pos]
                   << " ,actuality:" << sub_field;
        return false;
      }

      dtype = "";
      if (obj.at(pos).isString()) {
        dtype = obj.at(pos).getString(NULL);
      }
      if (SCM_VTYPE == sub_field || SCM_ETYPE == sub_field ||
          SCM_FIELD_CAPACITY == sub_field || SCM_FIELD_NAME == sub_field) {
        if (std::string::npos != dtype.find("DT")) {
          LOG(ERROR) << " The field value should be number!"
                     << " field:" << sub_field;
          return false;
        }
        ++pos;
        ++pre_pos;
        continue;
      } else if (unlikely(SCM_ATTRS == sub_field)) {
        this->CheckVertexOrEdgeValidity(obj[sub_field], sub_field);
        ++pos;
        ++pre_pos;
        continue;
      } else if (sub_field.find(SCM_ENTITY) != std::string::npos) {
        if (dtype != "DT_INT64") {
          LOG(ERROR) << " The field dtype is invalid!"
                     << " field:" << sub_field << " ,actual_dtype:" << dtype
                     << " ,expected_dtype:DT_INT64";
          return false;
        }
      } else if (sub_field.find(SCM_WEIGHT) != std::string::npos) {
        if (dtype != "DT_FLOAT" && dtype != "DT_DOUBLE") {
          LOG(ERROR) << " The field dtype is invalid!"
                     << " field:" << sub_field << " ,actual_dtype:" << dtype
                     << " ,expected_dtype:DT_FLOAT|DT_DOUBLE";
          return false;
        }
      } else if (sub_field == "type_1" || sub_field == "type_2") {
        if (dtype != "DT_UINT8") {
          LOG(ERROR) << " The field dtype is invalid!"
                     << " field:" << sub_field << " ,actual_dtype:" << dtype
                     << " ,expected_dtype:DT_UINT8";
          return false;
        }
      }

      if (unlikely(!DataType_Parse(dtype, &enum_dt))) {
        LOG(ERROR) << " The field dtype is invalid!"
                   << " field:" << sub_field << " ,dtype:" << dtype;
        return false;
      }

      ++pos;
      ++pre_pos;
    }
  }

  return true;
}

bool SchemaChecker::_InitPredefinedFields() {
  // The order of the fields are fixed in the schema.json
  main_fields_.emplace_back("vertexes");
  main_fields_.emplace_back("edges");

  std::vector<std::string> vertex_fields;
  vertex_fields.emplace_back(SCM_VTYPE);
  vertex_fields.emplace_back(SCM_ENTITY);
  vertex_fields.emplace_back(SCM_WEIGHT);
  vertex_fields.emplace_back(SCM_ATTRS);

  std::vector<std::string> edge_fields;
  edge_fields.emplace_back(SCM_ETYPE);
  edge_fields.emplace_back(SCM_ENTITY_1);
  edge_fields.emplace_back(SCM_ENTITY_2);
  edge_fields.emplace_back(SCM_WEIGHT);
  edge_fields.emplace_back(SCM_ATTRS);

  std::vector<std::string> attr_fields;
  attr_fields.emplace_back(SCM_FIELD_NAME);
  attr_fields.emplace_back(SCM_FIELD_TYPE);
  attr_fields.emplace_back(SCM_FIELD_CAPACITY);

  sub_fields_.emplace("vertex", std::move(vertex_fields));
  sub_fields_.emplace("edge", std::move(edge_fields));
  sub_fields_.emplace("attrs", std::move(attr_fields));

  return true;
}

const std::vector<std::string>& SchemaChecker::_GetPredefinedSubFieldsName(
    const std::string& main_field) {
  SchemaDict::iterator iter = sub_fields_.find(main_field);
  if (unlikely(iter == sub_fields_.end())) {
    static std::vector<std::string> pp;
    return pp;
  }

  return iter->second;
}

}  // namespace schema
}  // namespace galileo
