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
#include <vector>

#include "common/schema.h"
#include "utils/buffer.h"

namespace galileo {
namespace convertor {

using Buffer = galileo::utils::Buffer;
using Schema = galileo::schema::Schema;

struct Record {
  Buffer fix_fields;
  Buffer vary_fields;
  Buffer vary_state;
  void clear() {
    fix_fields.clear();
    vary_fields.clear();
    vary_state.clear();
  }
};

class TransformHelp {
 public:
  TransformHelp() {}
  ~TransformHelp() {}

 public:
  static bool TransformVertex(const char* line, size_t len, char field_split,
                              char array_split, Schema& schema,
                              Buffer& out_buff);
  static bool TransformEdge(const char* line, size_t len, char field_split,
                            char array_split, Schema& schema, Buffer& out_buff);

  static bool TransformVertexForUpdate(const char* line, size_t len,
                                       char field_split, char array_split,
                                       Schema& schema, Buffer& out_buff);
  static bool TransformEdgeForUpdate(const char* line, size_t len,
                                     char field_split, char array_split,
                                     Schema& schema, Buffer& out_buff);

  static bool TransformVertex(Schema& schema,
                              std::vector<std::vector<char*>>& fields,
                              Record& record);
  static bool TransformEdge(Schema& schema,
                            std::vector<std::vector<char*>>& fields,
                            Record& record);

  static int GetSliceId(const char* entity, std::string& e_dtype,
                        int partitions);
  static size_t SplitLine(char* line, size_t len, const char field_split,
                          const char array_split,
                          std::vector<std::vector<char*>>& fields);

 private:
  static bool _ParseTypeField(int type, Buffer* fields);
  static size_t _ParseField(char* data, size_t len, std::string& dtype,
                            Buffer* fields);

  static void _ParseState(Buffer* fields,
                          std::vector<uint8_t>& var_field_states);
  static void _CheckState(std::string& e_dtype,
                          std::vector<uint8_t>& var_field_states,
                          size_t data_len);
  static bool _TransformRecord(
      std::vector<std::vector<char*>>& fields, Record& record,
      std::function<std::string(uint8_t, size_t)> get_field_type_fun);
  static bool _GrantRecord(Record& record, Buffer& out_buff);
  static bool _GrantRecordForUpdate(Record& record, Buffer& out_buff);
  static size_t _SplitProperty(char* line, size_t len, const char array_split,
                               std::vector<char*>& elements);
  static bool _TransformProperty(std::string& dtype,
                                 std::vector<char*>& elements, Buffer& out_buff,
                                 size_t& data_len);
};

}  // namespace convertor
}  // namespace galileo
