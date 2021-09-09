# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

all_vertex_types = [[1000, 1002, 1003, 1004, 1005],
                    [1001, 1006, 1007, 1008, 1009, 1010]]
all_vertex = all_vertex_types[0] + all_vertex_types[1]
all_edges_types = [[[1000, 1001], [1001, 1003], [1001, 1004], [1001, 1005],
                    [1002, 1001]],
                   [[1001, 1000], [1006, 1000], [1007, 1000], [1008, 1000],
                    [1009, 1000], [1010, 1000]]]
all_edges = all_edges_types[0] + all_edges_types[1]
