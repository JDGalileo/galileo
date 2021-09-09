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

from galileo.platform.export import export
from . import data_source, utils


@export()
def get_data_source(data_source_name, output_path='./.data', **kwargs):
    r'''
    Download data source and convert

    args:
        data_source_name: ppi, cora, citeseer, pubmed
        output_path: root outpur path
        slice_count: partition count
        worker_count: worker count for convert
    '''
    data_source_name = data_source_name.lower()
    if data_source_name == 'ppi':
        from .ppi import PPI
        return PPI(output_path, data_source_name, **kwargs)
    elif data_source_name in ['cora', 'citeseer', 'pubmed']:
        from .planetoid import Planetoid
        return Planetoid(output_path, data_source_name, **kwargs)
    else:
        raise RuntimeError(f'not support data source {data_source_name}')


@export()
def get_evaluate_vertex_ids(data_source_name, **kwargs):
    r'''
    get evaluate vertex ids for data source name

    args:
        data_source_name: ppi, cora, citeseer, pubmed
        output_path: root outpur path
        slice_count: partition count
        worker_count: worker count for convert

    return:
        numpy array
    '''
    ds = get_data_source(data_source_name, **kwargs)
    return ds.get_evaluate_vertex_ids()


@export()
def get_test_vertex_ids(data_source_name, **kwargs):
    r'''
    get test vertex ids for data source name

    args:
        data_source_name: ppi, cora, citeseer, pubmed
        output_path: root outpur path
        slice_count: partition count
        worker_count: worker count for convert

    return:
        numpy array
    '''
    ds = get_data_source(data_source_name, **kwargs)
    return ds.get_test_vertex_ids()
