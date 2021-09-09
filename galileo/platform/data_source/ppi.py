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
import numpy as np
import zipfile
import networkx
from networkx.readwrite import json_graph
from galileo.platform.data_source.data_source import DataSource
from galileo.platform.data_source.utils import download_url
from galileo.platform.log import log

if networkx.__version__ != '2.3':
    raise RuntimeError('please use networkx version 2.3')


class PPI(DataSource):
    url = 'http://snap.stanford.edu/graphsage'

    def __init__(self, root_dir, name, **kwargs):
        self.name = name.lower()
        super().__init__(root_dir, name, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root_dir, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['ppi.zip']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)
        for name in self.raw_file_names:
            with zipfile.ZipFile(os.path.join(self.raw_dir, name)) as ppi_zip:
                ppi_zip.extractall(self.raw_dir)
        log.info('download ppi done')

    def read_data(self, prefix, normalize=True):
        G = json_graph.node_link_graph(json.load(open(prefix + "-G.json")))
        if os.path.exists(prefix + "-feats.npy"):
            feats = np.load(prefix + "-feats.npy")
        else:
            print("No features present.. Only identity features will be used.")
            feats = None
        id_map = json.load(open(prefix + "-id_map.json"))
        change_key = lambda n: int(n)
        id_map = {change_key(k): int(v) for k, v in id_map.items()}
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            change_lab = lambda n: n
        else:
            change_lab = lambda n: int(n)

        class_map = {
            change_key(k): change_lab(v)
            for k, v in class_map.items()
        }
        for node in G.nodes():
            if not 'val' in G.node[node] or not 'test' in G.node[node]:
                G.remove_node(node)
            else:
                if G.node[node]['val']:
                    G.node[node]['node_type'] = 0
                elif G.node[node]['test']:
                    G.node[node]['node_type'] = 1
                else:
                    G.node[node]['node_type'] = 2

        for edge in G.edges():
            if (G.node[edge[0]]['val'] or G.node[edge[1]]['val']
                    or G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                G[edge[0]][edge[1]]['edge_type'] = 0
            else:
                G[edge[0]][edge[1]]['edge_type'] = 1

        if normalize and not feats is None:
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([
                id_map[n] for n in G.nodes()
                if not G.node[n]['val'] and not G.node[n]['test']
            ])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)

        self.evaluate_vertex_ids = tuple(range(44906, 51420))
        self.test_vertex_ids = tuple(range(51420, 56944))
        return G, feats, id_map, class_map

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
                    "name": "feature1",
                    "dtype": "DT_ARRAY_INT32"
                }, {
                    "name": "feature2",
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
                    "name": "feature1",
                    "dtype": "DT_ARRAY_INT32"
                }, {
                    "name": "feature2",
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
                    "name": "feature1",
                    "dtype": "DT_ARRAY_INT32"
                }, {
                    "name": "feature2",
                    "dtype": "DT_ARRAY_FLOAT"
                }]
            }],
            "edges": [{
                "etype": 0,
                "entity_1": "DT_INT64",
                "entity_2": "DT_INT64",
                "weight": "DT_FLOAT",
                "attrs": []
            }, {
                "etype": 1,
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
        prefix = os.path.join(self.raw_dir, "ppi/ppi")
        G, feats, id_map, class_map = self.read_data(prefix)

        out_vertex = open(self.vertex_txt_path, 'w')
        out_edge = open(self.edge_txt_path, 'w')
        for node in G.nodes():
            node_type = G.node[node]['node_type']
            node_id = node
            node_weight = 1
            feature1 = class_map[node]
            feature1 = ','.join(str(x) for x in feature1)
            feature2 = list(feats[node])
            feature2 = ','.join(str(x) for x in feature2)
            out_vertex.write(
                f"{node_type}\t{node_id}\t{node_weight}\t{feature1}\t{feature2}\n"
            )
            for tar in G[node]:
                edge_type = G[node][tar]['edge_type']
                src_id = node
                dst_id = tar
                edge_weight = 1
                out_edge.write(
                    f"{edge_type}\t{src_id}\t{dst_id}\t{edge_weight}\n")

        out_vertex.close()
        out_edge.close()
        log.info(f'convert {self.name} to graph txt files done')
