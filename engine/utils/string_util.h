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

#include <sstream>
#include <string>
#include <vector>

namespace galileo {
namespace utils {

size_t split_string(const std::string& s, char delim,
                    std::vector<std::string>* v);
std::string join_string(const std::vector<std::string>& parts,
                        const std::string& separator);
std::string& ltrim(const std::string& chars, std::string& str);
std::string& rtrim(const std::string& chars, std::string& str);
std::string& trim(const std::string& chars, std::string& str);

// can not transform uint8_t and int8_t
template <typename Type>
Type strToNum(const char* str) {
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}

uint8_t strToUInt8(const char* str);
int8_t strToInt8(const char* str);
bool endswith(const char* s1, const char* s2);

}  // namespace utils
}  // namespace galileo
