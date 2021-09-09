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

#include "vertex_worker.h"
#include <stdlib.h>
#include <string>
#include "converter.h"
#include "convertor/transform_help.h"
#include "utils/string_util.h"

#include "glog/logging.h"

namespace galileo {
namespace convertor {

AllocIdManager VertexWorker::alloc_id_manager_;
bool VertexWorker::ParseRecord(std::vector<std::vector<char*>>& fields) {
  uint8_t vtype = galileo::utils::strToUInt8(fields[0][0]);

  int tmp_idx = converter_->schema_.GetVFieldIdx(vtype, SCM_ENTITY);
  if (tmp_idx < 0) {
    LOG(ERROR) << "get vertex entity field idx fail.vtype:"
               << std::to_string(vtype);
    return false;
  }
  size_t entity_idx = static_cast<size_t>(tmp_idx);
  assert(1 == fields[entity_idx].size());
  if (1 != fields[entity_idx].size()) {
    LOG(ERROR) << "entity_idx error:" << fields[entity_idx].size();
    return false;
  }
  char* entity = fields[entity_idx][0];
  std::string entity_dtype =
      converter_->schema_.GetVFieldDtype(vtype, entity_idx);
  int partitions = converter_->slice_count_;
  int slice_id = TransformHelp::GetSliceId(entity, entity_dtype, partitions);
  if (slice_id < 0) {
    LOG(ERROR) << "get the vertex slice id fail,the entity dtype is:"
               << entity_dtype;
    return false;
  }

  if (!TransformHelp::TransformVertex(converter_->schema_, fields, record_)) {
    return false;
  }

  return this->WriteRecord(slice_id, record_);
}

}  // namespace convertor
}  // namespace galileo
