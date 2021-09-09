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

import numpy as np
from galileo.platform.export import export


@export()
def get_fanouts_list(fanouts: list):
    r'''
    \brief cumprod of fanouts

    \param fanouts
    \return fanouts list
    '''
    return np.array([1] + fanouts).cumprod().tolist()


@export()
def get_fanouts_dim(fanouts: list):
    r'''
    \brief dim of fanouts list

    \param fanouts
    \return total dim
    '''
    return sum(get_fanouts_list(fanouts))


@export()
def get_fanouts_indices(fanouts: list):
    r'''
    \brief generate indices for relation

    \par Examples:
    \code{.py}
    In [1]: from galileo import get_fanouts_indices

    In [2]: get_fanouts_indices([2,3])
    Out[2]: [0, 1, 0, 2, 1, 3, 1, 4, 1, 5, 2, 6, 2, 7, 2, 8]

    In [3]: get_fanouts_indices([3,2])
    Out[3]: [0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2, 7, 3, 8, 3, 9]
    \endcode
    '''
    fanouts_list = get_fanouts_list(fanouts)
    fanouts_indices_list = fanouts_list[:-1]
    fanouts_indices_dim = sum(fanouts_indices_list)
    src_i = np.repeat(np.arange(fanouts_indices_dim),
                      np.repeat(fanouts, fanouts_indices_list))
    dst_i = np.arange(1, np.size(src_i) + 1)
    indices = np.reshape(np.stack([src_i, dst_i], axis=1), [-1])
    return indices.tolist()
