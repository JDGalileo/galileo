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

#include <string>
#include <unordered_map>
#include <vector>

#include "quickjson/value.h"

namespace galileo {
namespace schema {

class SchemaChecker {
  using SchemaDict = std::unordered_map<std::string, std::vector<std::string>>;

 public:
  SchemaChecker();
  ~SchemaChecker();

 public:
  bool CheckFirstLevelValidity(quickjson::Value& root);
  bool CheckVertexOrEdgeValidity(const quickjson::Value& values,
                                 const std::string& main_field);

 private:
  bool _InitPredefinedFields();
  const std::vector<std::string>& _GetPredefinedSubFieldsName(
      const std::string& main_field);

 private:
  std::vector<std::string> main_fields_;  // first level field
  SchemaDict sub_fields_;
};

}  // namespace schema
}  // namespace galileo
