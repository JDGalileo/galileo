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

import os
import json
import pickle
import numpy as np
import scipy.sparse as sp
from galileo.platform.data_source.data_source import DataSource
from galileo.platform.data_source.utils import download_url
from galileo.platform.log import log


class Planetoid(DataSource):
    r'''
    The citation network datasets 'Cora', 'CiteSeer' and 'PubMed'
    from 'Revisiting Semi-Supervised Learning with Graph Embeddings'
    <https://arxiv.org/abs/1603.08861>

    Nodes represent documents and edges represent citation links.
    '''

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root_dir, name, **kwargs):
        super().__init__(root_dir, name, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root_dir, self.name, 'raw')

    @property
    def raw_file_names(self):
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name, name) for name in names]

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)
        log.info(f'download {self.name} done')

    def read_data(self):
        '''
        files:
        x: feature vectors of training
        y: one-hot labels of training
        tx: feature vectors of test
        ty: one-hot labels of test
        allx, ally
        graph: dict, neighbors of nodes
        test.index: the indices of test instances in graph
        '''
        data = []
        for path in self.raw_paths:
            if path.endswith('test.index'):
                data.append([int(line.strip()) for line in open(path)])
            else:
                with open(path, 'rb') as f:
                    data.append(pickle.load(f, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph, test_idx = tuple(data)
        test_idx_range = np.sort(test_idx)

        if self.name == 'citeseer':
            # There are some isolated nodes in the Citeseer graph,
            # resulting in none consecutive test indices.
            # We need to identify them and add them as zero vectors
            # to `tx` and `ty`.
            min_test_idx = min(test_idx)
            len_test_idx = max(test_idx) - min_test_idx + 1
            tx_extended = sp.lil_matrix((len_test_idx, tx.shape[1]))
            ty_extended = np.zeros((len_test_idx, ty.shape[1]))
            tx_extended[test_idx_range - min_test_idx, :] = tx
            ty_extended[test_idx_range - min_test_idx, :] = ty
            tx, ty = tx_extended, ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx, :] = features[test_idx_range, :]
        features = features.todense()
        labels = np.vstack((ally, ty))
        labels[test_idx, :] = labels[test_idx_range, :]
        labels = labels.astype(float)
        assert features.shape[0] == labels.shape[0]

        train_idx = range(len(y))
        val_idx = range(len(y), len(y) + 500)
        test_idx = test_idx_range.tolist()
        mask = np.zeros(labels.shape[0], dtype=int)
        mask[train_idx] = 0
        mask[val_idx] = 1
        mask[test_idx] = 2

        self.evaluate_vertex_ids = tuple(val_idx)
        self.test_vertex_ids = tuple(test_idx)

        return features, labels, graph, mask

    def convert_to_schema(self):
        schema = {
            'vertexes': [{
                "vtype":
                0,
                "entity":
                "DT_INT64",
                "weight":
                "DT_FLOAT",
                "attrs": [{
                    "name": "label",
                    "dtype": "DT_ARRAY_FLOAT"
                }, {
                    "name": "feature",
                    "dtype": "DT_ARRAY_FLOAT"
                }]
            }, {
                "vtype":
                1,
                "entity":
                "DT_INT64",
                "weight":
                "DT_FLOAT",
                "attrs": [{
                    "name": "label",
                    "dtype": "DT_ARRAY_FLOAT"
                }, {
                    "name": "feature",
                    "dtype": "DT_ARRAY_FLOAT"
                }]
            }, {
                "vtype":
                2,
                "entity":
                "DT_INT64",
                "weight":
                "DT_FLOAT",
                "attrs": [{
                    "name": "label",
                    "dtype": "DT_ARRAY_FLOAT"
                }, {
                    "name": "feature",
                    "dtype": "DT_ARRAY_FLOAT"
                }]
            }],
            "edges": [{
                "etype": 0,
                "entity_1": "DT_INT64",
                "entity_2": "DT_INT64",
                "weight": "DT_FLOAT",
                "attrs": []
            }]
        }
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)
        log.info(f'write {self.name} schema done')

    def convert_to_txt(self):
        features, labels, graph, mask = self.read_data()
        samples = features.shape[0]
        with open(self.vertex_txt_path, 'w') as f:
            for i in range(samples):
                f.write(
                    f"{mask[i]}\t{i}\t1\t{','.join(str(x) for x in labels[i].tolist())}\t"
                    f"{','.join(str(x) for x in features[i].A1.tolist())}\n")

        with open(self.edge_txt_path, 'w') as f:
            for src, dst in graph.items():
                for d in dst:
                    f.write(f'0\t{src}\t{d}\t1\n')
            for i in range(samples):
                f.write(f'0\t{i}\t{i}\t1\n')
        log.info(f'convert {self.name} to graph txt files done')
