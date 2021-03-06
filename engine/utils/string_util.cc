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

#include "utils/string_util.h"

#include <cstdlib>
#include <cstring>

namespace galileo {
namespace utils {

std::string& ltrim(const std::string& chars, std::string& str) {
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

std::string& rtrim(const std::string& chars, std::string& str) {
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

std::string& trim(const std::string& chars, std::string& str) {
  return ltrim(chars, rtrim(chars, str));
}

size_t split_string(const std::string& s, char delim,
                    std::vector<std::string>* v) {
  if (v == NULL) return 0;
  v->clear();
  size_t pos = s.find(delim);
  size_t beg = 0;
  while (pos != std::string::npos) {
    v->push_back(std::string(s, beg, pos - beg));
    beg = pos + 1;
    pos = s.find(delim, beg);
  }
  v->push_back(std::string(s, beg));
  return v->size();
}

std::string join_string(const std::vector<std::string>& parts,
                        const std::string& separator) {
  std::stringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << separator;
    }
    ss << parts[i];
  }
  return ss.str();
}

bool endswith(const char* s1, const char* s2) {
  size_t n1 = std::strlen(s1);
  size_t n2 = std::strlen(s2);
  if (n1 < n2) return false;
  if (!n2) return true;
  const char* e1 = s1 + n1 - 1;
  const char* e2 = s2 + n2 - 1;
  char c1, c2;
  while (1) {
    if (e1 < s1 || e2 < s2) break;
    c1 = *e1--;
    c2 = *e2--;
    if (c1 != c2) return false;
  }
  if (e2 >= s2) return false;
  return true;
}

uint8_t strToUInt8(const char* str) {
  int tmp = std::atoi(str);
  uint8_t num = static_cast<uint8_t>(tmp);
  return num;
}

int8_t strToInt8(const char* str) {
  int tmp = std::atoi(str);
  int8_t num = static_cast<int8_t>(tmp);
  return num;
}

}  // namespace utils
}  // namespace galileo
