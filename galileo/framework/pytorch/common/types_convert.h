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

#ifndef __Types_Convert_H__
#define __Types_Convert_H__

#include <torch/extension.h>

#include "engine/client/dgraph_type.h"

namespace torch {
namespace glo {

inline Dtype EngineType2PT(int type) {
  switch (type) {
    case galileo::client::CT_BOOL:
      return kBool;
    case galileo::client::CT_UINT8:
      return kByte;
    case galileo::client::CT_INT8:
      return kChar;
    case galileo::client::CT_UINT16:
      return kShort;  // Undefined;
    case galileo::client::CT_INT16:
      return kShort;
    case galileo::client::CT_UINT32:
      return kInt;  // Undefined;
    case galileo::client::CT_INT32:
      return kInt;
    case galileo::client::CT_UINT64:
      return kLong;  // Undefined;
    case galileo::client::CT_INT64:
      return kLong;
    case galileo::client::CT_FLOAT:
      return kFloat;
    case galileo::client::CT_DOUBLE:
      return kDouble;
    case galileo::client::CT_STRING:
      return Dtype::Undefined;
    default:
      return Dtype::Undefined;
  }
}

}  // namespace glo
}  // namespace torch

#endif  // __Types_Convert_H__
