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

import torch
from galileo.platform.export import export

__ops_lib = None


def _get_ops_lib():
    global __ops_lib
    if __ops_lib is None:
        import galileo.framework.pywrap.py_client
        import galileo.framework.pywrap.pt_ops as _ops_lib
        __ops_lib = _ops_lib
    return __ops_lib


def _to_tensor(inputs, dtype):
    if isinstance(inputs, (list, tuple)):
        return torch.tensor(inputs, dtype=dtype)
    if torch.is_tensor(inputs) and inputs.dtype != dtype:
        return inputs.to(dtype=dtype)
    return inputs


def _to_long_tensor(inputs):
    return _to_tensor(inputs, torch.int64)


def _to_byte_tensor(inputs):
    return _to_tensor(inputs, torch.uint8)


class PTOps(object):
    r'''
    pytorch galileo ops
    '''
    @staticmethod
    def sample_vertices(types, count):
        r'''
        sample vertices

        Args:
          types: list[uint8_t], vertex types
          count: int, count per type
        Return:
          list[Tensor]
        '''
        return PTOps._collect_entity(types, count, category='vertex')

    @staticmethod
    def sample_edges(types, count):
        r'''
        sample edges

        Args:
            types: list[uint8_t], vertex types
            count: int, count per type
        Return:
            list[Tensor]
        '''
        return PTOps._collect_entity(types, count, category='edge')

    @staticmethod
    def sample_neighbors(vertices, edge_types, count, has_weight=False):
        r'''
        sample neighbors

        Args:
          vertices: list[int] or torch.LongTensor, vertices
          edge_types: list[int] or torch.ByteTensor, edge type
          count: int, neighbor per vertices
          has_weight: bool, whether output weight

        return:
          list[torch.Tensor]
        '''
        vertices = _to_long_tensor(vertices)
        edge_types = _to_byte_tensor(edge_types)
        return _get_ops_lib().collect_state_neighbor(vertices, edge_types,
                                                     count, has_weight)

    @staticmethod
    def get_topk_neighbors(vertices, edge_types, k, has_weight=False):
        r'''
        get topk neighbors

        Args:
            vertices: list[int64_t] vertices
            edge_types: list[uint8_t] edge types
            k: int, k of topk neighbor per vertex
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        return PTOps._collect_neighbor(vertices, edge_types, k, has_weight,
                                       'topk')

    @staticmethod
    def get_full_neighbors(vertices, edge_types, has_weight=False):
        r'''
        get full neighbors

        Args:
            vertices: list[int64_t] vertices
            edge_types: list[uint8_t] edge types
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        return PTOps._collect_neighbor(vertices, edge_types, 0, has_weight,
                                       'full')

    @staticmethod
    def get_pod_feature(ids, fnames, dims, ftypes):
        r'''
        collect pod feature

        Args:
          ids: list[list[int]] or list[Tensor], vertex or edge
          fnames: list[string], feature name
          dims: list[int], dims
          ftypes:list[torch_type],output type
        return:
          list[torch.Tensor]
        '''
        for idx, val in enumerate(ids):
            if idx == 0 or idx == 1:
                ids[idx] = _to_long_tensor(val)
            elif idx == 2:
                ids[idx] = _to_byte_tensor(val)
        return _get_ops_lib().collect_pod_feature(ids, fnames, dims)

    @staticmethod
    def sample_seq_by_multi_hop(vertices, metapath, fanouts, has_weight=False):
        r'''
        sample sequence multi hops, including vertices

        Args:
            vertices: list[int64_t], vertices
            metapath: list[list[uint8_t]] edge types for every hop
            fanouts: list[int], fanouts for every hop
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        vertices = _to_long_tensor(vertices)
        for idx, val in enumerate(metapath):
            metapath[idx] = _to_byte_tensor(val)
        return _get_ops_lib().collect_seq_by_multi_hop(vertices, metapath,
                                                       fanouts, has_weight)

    @staticmethod
    def sample_seq_by_random_walk(vertices,
                                  metapath,
                                  repetition,
                                  p=1.0,
                                  q=1.0):
        r'''
        sample sequence random walk

        Args:
            vertices: list[int64_t], vertices
            metapath: list[list[uint8_t]], edge types per walk
            repetition: int, walks per vertex
            p: float, return parameter, default is 1.0
            q: float, in-out parameter, default is 1.0
        Return:
            Tensor
        '''
        vertices = _to_long_tensor(vertices)
        for idx, val in enumerate(metapath):
            metapath[idx] = _to_byte_tensor(val)
        return _get_ops_lib().collect_seq_by_rw_with_bias(
            vertices, metapath, repetition, p, q)

    @staticmethod
    def sample_pairs_by_random_walk(vertices,
                                    metapath,
                                    repetition,
                                    context_size,
                                    p=1.0,
                                    q=1.0):
        r'''
        sample pairs random walk

        Args:
            vertices: list[int64_t], vertices
            metapath: list[list[uint8_t]], edge types per walk
            repetition: int, walks per vertex
            context_size: int, the context size of pairs
            p: float, return parameter, default is 1.0
            q: float, in-out parameter, default is 1.0
        Return:
            Tensor
        '''
        vertices = _to_long_tensor(vertices)
        for idx, val in enumerate(metapath):
            metapath[idx] = _to_byte_tensor(val)
        return _get_ops_lib().collect_pair_by_rw_with_bias(
            torch.LongTensor(vertices),
            metapath,
            repetition,
            context_size,
            p,
            q,
        )

    @staticmethod
    def _collect_entity(types, count, category):
        r'''
        collect entity, eg vertex, edge.

        Args:
          types: list[int] or torch.ByteTensor, entity types
          count: int, entity count per type
          category: str, vertex or edge

        return:
          list[torch.Tensor]
        '''
        types = _to_byte_tensor(types)
        return _get_ops_lib().collect_entity(types, count, category)

    @staticmethod
    def _collect_neighbor(vertices, edge_types, count, has_weight, category):
        r'''
        collect neighbor

        Args:
          vertices: list[int] or torch.LongTensor, vertices
          edge_types: list[int], edge type
          count: int, neighbor per vertices
          has_weight: bool, whether output weight
          category: str, topk or full

        return:
          list[torch.Tensor]
        '''
        vertices = _to_long_tensor(vertices)
        edge_types = _to_byte_tensor(edge_types)
        return _get_ops_lib().collect_neighbor(vertices, edge_types, count,
                                               has_weight, category)


export('galileo.pytorch').var('ops', PTOps)
