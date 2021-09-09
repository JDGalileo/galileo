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

import tensorflow as tf
from galileo.platform.export import export

__ops_lib = None


def _get_ops_lib():
    global __ops_lib
    if __ops_lib is None:
        import galileo.framework.pywrap.py_client
        from galileo.platform.path_helper import get_tf_ops
        __ops_lib = tf.load_op_library(get_tf_ops())
    return __ops_lib


def get_entity_dataset():
    return _get_ops_lib().entity_dataset


class TFOps(object):
    r'''
    tensorflow galileo ops
    '''
    @staticmethod
    def sample_vertices(types, count):
        r'''
        sample vertices

        Args:
            types: list[int] or tf.Tensor(dtype=uint8), vertex types
            count: int or tf.Tensor, count per type
        Return:
            list[Tensor]
        '''
        return _get_ops_lib().collect_entity(types,
                                             count,
                                             category='vertex',
                                             T=[tf.int64])

    def sample_edges(types, count):
        r'''
        sample edges

        Args:
            types: list[int] or tf.Tensor(dtype=uint8), vertex types
            count: int or tf.Tensor, count per type
        Return:
            list[Tensor]
        '''
        return _get_ops_lib().collect_entity(types,
                                             count,
                                             category='edge',
                                             T=[tf.int64, tf.int64, tf.uint8])

    @staticmethod
    def sample_neighbors(vertices, edge_types, count, has_weight=False):
        r'''
        sample neighbors

        Args:
            vertices: list[int] or tf.Tensor(dtype=int64), vertices
            edge_types: list[int] or tf.Tensor(dtype=uint8), edge type
            count: int, neighbor per vertices
            has_weight: bool, whether output weight

        return:
            list[Tensor]
        '''
        types = [tf.int64, tf.float32] if has_weight else [tf.int64]
        return _get_ops_lib().collect_state_neighbor(vertices,
                                                     edge_types,
                                                     count=count,
                                                     T=types)

    @staticmethod
    def get_topk_neighbors(vertices, edge_types, k, has_weight=False):
        r'''
        get topk neighbors

        Args:
            vertices: list[int] vertices
            edge_types: list[int] edge types
            k: int, k of topk neighbor per vertex
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        types = [tf.int64, tf.float32] if has_weight else [tf.int64]
        return _get_ops_lib().collect_neighbor(vertices,
                                               edge_types,
                                               count=k,
                                               category='topk',
                                               T=types)

    @staticmethod
    def get_full_neighbors(vertices, edge_types, has_weight=False):
        r'''
        get full neighbors

        Args:
            vertices: list[int] vertices
            edge_types: list[int] edge types
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        types = ([tf.int64, tf.float32, tf.int32]
                 if has_weight else [tf.int64, tf.int32])
        return _get_ops_lib().collect_neighbor(vertices,
                                               edge_types,
                                               count=0,
                                               category='full',
                                               T=types)

    @staticmethod
    def get_pod_feature(ids, fnames, dims, ftypes):
        r'''
        collect pod feature

        Args:
            ids: list[list[int]] or list[Tensor], vertex or edge
            fnames: list[string], feature name
            dims: list[int], dims
            ftypes: list[tf_type], output type

        return
            list[tf.Tensor]
        '''
        for idx, val in enumerate(ids):
            if (idx == 0 or idx == 1) and not tf.is_tensor(val):
                ids[idx] = tf.convert_to_tensor(val, dtype=tf.int64)
            elif idx == 2 and not tf.is_tensor(val):
                ids[idx] = tf.convert_to_tensor(val, dtype=tf.uint8)

        return _get_ops_lib().collect_feature(ids,
                                              fnames=fnames,
                                              dimensions=dims,
                                              TO=ftypes)

    @staticmethod
    def sample_seq_by_multi_hop(vertices, metapath, fanouts, has_weight=False):
        r'''
        sample sequence multi hops, including vertices

        Args:
            vertices: list[int] or Tensor(dtype=int64), vertices
            metapath: list[list[int]] or list[Tensor(dtype=uint8)],
                edge types for every hop
            fanouts: list[int], count for every hop
            has_weight: bool, whether output weight
        Return:
            list[Tensor]
        '''
        if tf.is_tensor(vertices):
            vertices = tf.reshape(vertices, [-1])
        types = [tf.int64, tf.float32] if has_weight else [tf.int64]
        return _get_ops_lib().collect_seq_by_multi_hop(vertices,
                                                       metapath,
                                                       counts=fanouts,
                                                       T=types)

    @staticmethod
    def sample_seq_by_random_walk(vertices,
                                  metapath,
                                  repetition,
                                  p=1.0,
                                  q=1.0):
        r'''
        sample sequence random walk

        Args:
            vertices: list[int] or Tensor(dtype=int64), vertices
            metapath: list[list[int]] or list[Tensor(dtype=uint8)]
                edge types per walk
            repetition: int, walks per vertex
            p: float, return parameter, default is 1.0
            q: float, in-out parameter, default is 1.0
        Return:
            Tensor
        '''
        return _get_ops_lib().collect_seq_by_rw_with_bias(
            vertices, metapath, repetition=repetition, p=p, q=q)

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
            vertices: list[int] or Tensor(dtype=int64), vertices
            metapath: list[list[int]] or list[Tensor(dtype=uint8)]
                edge types per walk
            repetition: int, walks per vertex
            context_size: int, the context size of pairs
            p: float, return parameter, default is 1.0
            q: float, in-out parameter, default is 1.0
        Return:
            Tensor
        '''
        return _get_ops_lib().collect_pair_by_rw_with_bias(
            vertices,
            metapath,
            repetition=repetition,
            p=p,
            q=q,
            context_size=context_size)


export('galileo.tf').var('ops', TFOps)
