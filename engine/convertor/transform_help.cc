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

#include "convertor/transform_help.h"
#include "common/macro.h"
#include "common/types.h"
#include "glog/logging.h"
#include "utils/string_util.h"

namespace galileo {
namespace convertor {

bool TransformHelp::TransformVertex(const char* line, size_t len,
                                    char field_split, char array_split,
                                    Schema& schema, Buffer& out_buff) {
  std::vector<std::vector<char*>> fields;
  SplitLine((char*)line, len, field_split, array_split, fields);
  Record record;
  if (!TransformVertex(schema, fields, record)) {
    return false;
  }
  return _GrantRecord(record, out_buff);
}
bool TransformHelp::TransformEdge(const char* line, size_t len,
                                  char field_split, char array_split,
                                  Schema& schema, Buffer& out_buff) {
  std::vector<std::vector<char*>> fields;
  SplitLine((char*)line, len, field_split, array_split, fields);
  Record record;
  if (!TransformEdge(schema, fields, record)) {
    return false;
  }
  return _GrantRecord(record, out_buff);
}
bool TransformHelp::TransformVertexForUpdate(const char* line, size_t len,
                                             char field_split, char array_split,
                                             Schema& schema, Buffer& out_buff) {
  std::vector<std::vector<char*>> fields;
  SplitLine((char*)line, len, field_split, array_split, fields);
  Record record;
  if (!TransformVertex(schema, fields, record)) {
    return false;
  }
  return _GrantRecordForUpdate(record, out_buff);
}
bool TransformHelp::TransformEdgeForUpdate(const char* line, size_t len,
                                           char field_split, char array_split,
                                           Schema& schema, Buffer& out_buff) {
  std::vector<std::vector<char*>> fields;
  SplitLine((char*)line, len, field_split, array_split, fields);
  Record record;
  if (!TransformEdge(schema, fields, record)) {
    return false;
  }
  return _GrantRecordForUpdate(record, out_buff);
}
size_t TransformHelp::SplitLine(char* line, size_t len, const char field_split,
                                const char array_split,
                                std::vector<std::vector<char*>>& fields) {
  char* pc = line;
  char* pb = line;
  char* pe = pc + len;
  std::vector<char*> temp;
  char ch;
  while (ch = *pc, ch && pc < pe) {
    if (ch == array_split) {
      *pc = '\0';
      temp.emplace_back(pb);
      pc += 1;
      pb = pc;
    } else if (ch == field_split) {
      *pc = '\0';
      temp.emplace_back(pb);
      fields.emplace_back(temp);
      temp.clear();
      pc += 1;
      pb = pc;
    } else {
      pc += 1;
    }
  }
  if (pc > pb) {
    *pc = '\0';
    temp.emplace_back(pb);
    fields.emplace_back(temp);
  }

  return fields.size();
}

bool TransformHelp::TransformVertex(Schema& schema,
                                    std::vector<std::vector<char*>>& fields,
                                    Record& record) {
  std::function<std::string(uint8_t, size_t)> get_field_type_fun =
      std::bind(&Schema::GetVFieldDtype, &schema, std::placeholders::_1,
                std::placeholders::_2);
  uint8_t vtype = galileo::utils::strToUInt8(fields[0][0]);
  if (fields.size() != schema.GetVFieldCount(vtype)) {
    LOG(ERROR) << "the size of source data error,the entity vtype is:"
               << (uint32_t)vtype;
    return false;
  }
  return _TransformRecord(fields, record, get_field_type_fun);
}
bool TransformHelp::TransformEdge(Schema& schema,
                                  std::vector<std::vector<char*>>& fields,
                                  Record& record) {
  std::function<std::string(uint8_t, size_t)> get_field_type_fun =
      std::bind(&Schema::GetEFieldDtype, &schema, std::placeholders::_1,
                std::placeholders::_2);
  uint8_t etype = galileo::utils::strToUInt8(fields[0][0]);
  if (fields.size() != schema.GetEFieldCount(etype)) {
    LOG(ERROR) << "the size of source data error,the entity etype is:"
               << (uint32_t)etype;
    return false;
  }
  return _TransformRecord(fields, record, get_field_type_fun);
}

int TransformHelp::GetSliceId(const char* entity, std::string& e_dtype,
                              int partitions) {
  int slice_id = -1;
  if ("DT_INT64" == e_dtype) {
    int64_t entity_id = galileo::utils::strToNum<int64_t>(entity);
    slice_id = static_cast<int>(entity_id % partitions);
  } else {
    LOG(ERROR) << " the entity dtype is error."
               << " dtype:" << e_dtype;
    return -1;
  }

  return std::abs(slice_id);
}

bool TransformHelp::_ParseTypeField(int type, Buffer* fields) {
  if (type < 0 || type > galileo::common::MAX_8BIT_NUM) {
    return false;
  }

  uint8_t tmp_type = static_cast<uint8_t>(type);
  return fields->write((const char*)&tmp_type, sizeof(tmp_type));
}

size_t TransformHelp::_ParseField(char* data, size_t len, std::string& dtype,
                                  Buffer* fields) {
  if (dtype.find("UINT8") != std::string::npos) {
    uint8_t elem = galileo::utils::strToUInt8(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("INT8") != std::string::npos) {
    int8_t elem = galileo::utils::strToInt8(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("UINT16") != std::string::npos) {
    uint16_t elem = galileo::utils::strToNum<uint16_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("INT16") != std::string::npos) {
    int16_t elem = galileo::utils::strToNum<int16_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("UINT32") != std::string::npos) {
    uint32_t elem = galileo::utils::strToNum<uint32_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("INT32") != std::string::npos) {
    int32_t elem = galileo::utils::strToNum<int32_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("UINT64") != std::string::npos) {
    uint64_t elem = galileo::utils::strToNum<uint64_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("INT64") != std::string::npos) {
    int64_t elem = galileo::utils::strToNum<int64_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("FLOAT") != std::string::npos) {
    float elem = galileo::utils::strToNum<float>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("DOUBLE") != std::string::npos) {
    double elem = galileo::utils::strToNum<double>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("BOOL") != std::string::npos) {
    uint8_t elem = galileo::utils::strToNum<uint8_t>(data);
    if (!fields->write((const char*)&elem, sizeof(elem))) {
      return 0;
    }
    return sizeof(elem);
  } else if (dtype.find("STRING") != std::string::npos ||
             dtype.find("BINARY") != std::string::npos) {
    if (data[len - 1] == '\r') {
      --len;
    }
    if (!fields->write((const char*)&len, sizeof(uint16_t))) {
      return 0;
    }
    if (!fields->write(data, len)) {
      return 0;
    }
    return len;
  }

  LOG(ERROR) << " the dtype of field is error!"
             << " elem: " << data << " dtype: " << dtype;

  return 0;
}

void TransformHelp::_ParseState(Buffer* fields,
                                std::vector<uint8_t>& var_field_states) {
  if (var_field_states.size() <= 0) {
    return;
  }
  uint8_t remainder_bit_size =
      static_cast<uint8_t>(var_field_states.size() % 8);

  uint8_t state = 0;
  for (size_t i = 0; i < var_field_states.size(); ++i) {
    state = static_cast<uint8_t>((state << 1) | var_field_states[i]);
    if (i % 8 == 7) {
      fields->write((const char*)&state, sizeof(state));
      state = 0;
    }
  }
  if (remainder_bit_size > 0) {
    state = static_cast<uint8_t>(state << (8 - remainder_bit_size));
    fields->write((const char*)&state, sizeof(state));
  }
}

void TransformHelp::_CheckState(std::string& e_dtype,
                                std::vector<uint8_t>& var_field_states,
                                size_t data_len) {
  if (e_dtype.find("ARRAY") == std::string::npos &&
      e_dtype.find("STRING") == std::string::npos &&
      e_dtype.find("BINARY") == std::string::npos) {
    return;
  }
  if (data_len > 6) {
    var_field_states.push_back(0);
  } else {
    var_field_states.push_back(1);
  }
}
bool TransformHelp::_TransformRecord(
    std::vector<std::vector<char*>>& fields, Record& record,
    std::function<std::string(uint8_t, size_t)> get_field_type_fun) {
  uint8_t vtype = galileo::utils::strToUInt8(fields[0][0]);
  std::vector<uint8_t> var_field_states;
  record.clear();

  Buffer var_tmp_field;

  _ParseTypeField(vtype, &(record.fix_fields));
  for (size_t field_num = 1; field_num < fields.size(); ++field_num) {
    var_tmp_field.clear();
    size_t data_len = 0;
    std::string dtype = get_field_type_fun(vtype, field_num);
    Buffer* cur_fields = NULL;
    bool is_vary_field = false;
    if (dtype.find("ARRAY") != std::string::npos) {
      cur_fields = &(var_tmp_field);
      is_vary_field = true;
    } else if (dtype == "DT_STRING" || dtype == "DT_BINARY") {
      cur_fields = &(var_tmp_field);
      is_vary_field = true;
    } else {
      cur_fields = &(record.fix_fields);
    }

    if (!_TransformProperty(dtype, fields[field_num], *cur_fields, data_len)) {
      return false;
    }
    if (is_vary_field) {
      uint16_t vary_field_size = static_cast<uint16_t>(var_tmp_field.size());
      if (dtype.find("ARRAY") != std::string::npos) {
        record.vary_fields.write((const char*)&vary_field_size,
                                 sizeof(vary_field_size));
      }
      record.vary_fields.write(var_tmp_field.readBuffer(),
                               var_tmp_field.size());
    }
    _CheckState(dtype, var_field_states, data_len);
  }

  _ParseState(&(record.vary_state), var_field_states);
  return true;
}
bool TransformHelp::_GrantRecord(Record& record, Buffer& out_buff) {
  size_t fix_size = record.fix_fields.size();
  size_t vary_size = record.vary_fields.size();
  size_t vary_state_size = record.vary_state.size();
  // uint16_t record_size = fix_size + vary_state_size + vary_size;
  // if(unlikely(!out_buff.write((char*)&record_size, sizeof(record_size)))) {
  //     LOG(ERROR)<<" write record_size value fail!";
  //     return false;
  // }
  if (unlikely(!out_buff.write(record.fix_fields.buffer(), fix_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(!out_buff.write(record.vary_state.buffer(), vary_state_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(!out_buff.write(record.vary_fields.buffer(), vary_size))) {
    LOG(ERROR) << " write record vary field content fail!";
    return false;
  }

  return true;
}
bool TransformHelp::_GrantRecordForUpdate(Record& record, Buffer& out_buff) {
  size_t fix_size = record.fix_fields.size();
  size_t vary_size = record.vary_fields.size();
  size_t vary_state_size = record.vary_state.size();
  uint16_t record_size =
      static_cast<uint16_t>(fix_size + vary_state_size + vary_size);

  if (unlikely(!out_buff.write((char*)&record_size, sizeof(record_size)))) {
    LOG(ERROR) << " write record_size value fail!";
    return false;
  }
  if (unlikely(!out_buff.write(record.fix_fields.buffer(), fix_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(!out_buff.write(record.vary_state.buffer(), vary_state_size))) {
    LOG(ERROR) << " write record fix field content fail!";
    return false;
  }
  if (unlikely(!out_buff.write(record.vary_fields.buffer(), vary_size))) {
    LOG(ERROR) << " write record vary field content fail!";
    return false;
  }

  return true;
}

size_t TransformHelp::_SplitProperty(char* line, size_t len,
                                     const char array_split,
                                     std::vector<char*>& elements) {
  char* pc = line;
  char* pb = line;
  char* pe = pc + len;
  char ch;
  while (ch = *pc, ch && pc < pe) {
    if (ch == array_split) {
      *pc = '\0';
      elements.emplace_back(pb);
      pc += 1;
      pb = pc;
    } else {
      pc += 1;
    }
  }
  if (pc > pb) {
    *pc = '\0';
    elements.emplace_back(pb);
  }
  return elements.size();
}
bool TransformHelp::_TransformProperty(std::string& dtype,
                                       std::vector<char*>& elements,
                                       Buffer& out_buff, size_t& data_len) {
  uint16_t elem_count = static_cast<uint16_t>(elements.size());
  if (dtype.find("ARRAY") != std::string::npos) {
    out_buff.write((const char*)&elem_count, sizeof(elem_count));
  }

  for (size_t elem_num = 0; elem_num != elem_count; ++elem_num) {
    char* elem = elements[elem_num];
    size_t len = strlen(elem);
    size_t actual_len = _ParseField(elem, len, dtype, &out_buff);
    if (actual_len == 0) {
      LOG(ERROR) << " parse field fail!"
                 << " the elem: " << elem << " dtype: " << dtype;
      return false;
    }
    data_len += actual_len;
  }
  return true;
}

}  // namespace convertor
}  // namespace galileo
