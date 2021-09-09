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

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#ifdef client_EXPORTS
#define CLIENT_EXTERNAL __attribute__((visibility("default")))
#else
#define CLIENT_EXTERNAL
#endif  // client_EXPORTS

#define SCM_VTYPE "vtype"
#define SCM_ENTITY "entity"
#define SCM_WEIGHT "weight"
#define SCM_ATTRS "attrs"

#define SCM_ETYPE "etype"
#define SCM_ENTITY_1 "entity_1"
#define SCM_ENTITY_2 "entity_2"

#define SCM_FIELD_NAME "name"
#define SCM_FIELD_TYPE "dtype"
#define SCM_FIELD_CAPACITY "capacity"
