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

#include "engine/client/dgraph_type.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace glo {

inline DataType EngineType2TF(galileo::client::ClientType type) {
  switch (type) {
    case galileo::client::CT_BOOL:
      return DT_BOOL;
    case galileo::client::CT_UINT8:
      return DT_UINT8;
    case galileo::client::CT_INT8:
      return DT_INT8;
    case galileo::client::CT_UINT16:
      return DT_UINT16;
    case galileo::client::CT_INT16:
      return DT_INT16;
    case galileo::client::CT_UINT32:
      return DT_UINT32;
    case galileo::client::CT_INT32:
      return DT_INT32;
    case galileo::client::CT_UINT64:
      return DT_UINT64;
    case galileo::client::CT_INT64:
      return DT_INT64;
    case galileo::client::CT_FLOAT:
      return DT_FLOAT;
    case galileo::client::CT_DOUBLE:
      return DT_DOUBLE;
    case galileo::client::CT_STRING:
      return DT_STRING;
    default:
      return DT_INVALID;
  }
}

}  // namespace glo
}  // namespace tensorflow

#endif  // __Types_Convert_H__
