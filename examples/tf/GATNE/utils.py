# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
GATNE-T模型的utils
'''

import os
import json
import pickle
import numpy as np
import galileo as g
import galileo.tf as gt
from sklearn.metrics import f1_score, roc_auc_score


class DataSource(g.DataSource):
    r'''
    available dataset name: amazon example twitter youtube
    '''

    #url = 'https://github.com/THUDM/GATNE/tree/master/data'
    url = 'http://storage.jd.local/pinoctl/graphdata/GATNE'

    def __init__(self, name, root_dir='./.data', **kwargs):
        assert name.lower() in ('amazon', 'example', 'twitter', 'youtube')
        # download and convert in init
        super().__init__(root_dir, name, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root_dir, self.name, 'raw')

    @property
    def raw_file_names(self):
        names = ['train.txt', 'valid.txt', 'test.txt']
        if self.name in ('example', 'amazon'):
            names.append('feature.txt')
        return names

    def download(self):
        for name in self.raw_file_names:
            g.download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)
        print(f'download {self.name} done')

    # for each line, the data is [edge_type, node, node]
    def _load_training_data(self, f_name):
        edge_data_by_type = dict()
        all_nodes = list()
        with open(f_name, 'r') as f:
            for line in f:
                words = line.strip().split(' ')
                tp = int(words[0]) - 1
                if tp not in edge_data_by_type:
                    edge_data_by_type[tp] = list()
                x, y = words[1], words[2]
                edge_data_by_type[tp].append((x, y))
                all_nodes.append(x)
                all_nodes.append(y)
        id2index = dict()
        for i, v in enumerate(set(all_nodes)):
            id2index[v] = i
        return edge_data_by_type, id2index

    # for each line, the data is [edge_type, node, node, true_or_false]
    def _load_testing_data(self, f_name, id2index):
        true_edge_data_by_type = dict()
        false_edge_data_by_type = dict()
        with open(f_name, 'r') as f:
            for line in f:
                words = line.strip().split(' ')
                tp = int(words[0]) - 1
                x, y = id2index[words[1]], id2index[words[2]]
                if int(words[3]) == 1:
                    if tp not in true_edge_data_by_type:
                        true_edge_data_by_type[tp] = list()
                    true_edge_data_by_type[tp].append((x, y))
                else:
                    if tp not in false_edge_data_by_type:
                        false_edge_data_by_type[tp] = list()
                    false_edge_data_by_type[tp].append((x, y))
        return true_edge_data_by_type, false_edge_data_by_type

    def _load_feature_data(self, f_name):
        feature_dic = {}
        feature_dim = None
        with open(f_name, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                items = line.strip().split()
                feature_dic[items[0]] = items[1:]
                feature_dim_tmp = len(items[1:])
                if feature_dim is None:
                    feature_dim = feature_dim_tmp
                else:
                    if feature_dim != feature_dim_tmp:
                        raise ValueError('feature dim is not match '
                                         f'{feature_dim} vs {feature_dim_tmp} '
                                         f'for {items[0]}')
        return feature_dic, feature_dim

    def convert_to_schema(self, edge_type_num=None, has_feature=False):
        if edge_type_num is None:
            return
        edges = [{
            "etype": i,
            "entity_1": "DT_INT64",
            "entity_2": "DT_INT64",
            "weight": "DT_FLOAT",
            "attrs": []
        } for i in range(edge_type_num)]
        v_attrs = [{"name": "word", "dtype": "DT_INT64"}]
        if has_feature:
            v_attrs.append({"name": "feature", "dtype": "DT_ARRAY_FLOAT"})
        schema = {
            'vertexes': [{
                "vtype": 0,
                "entity": "DT_INT64",
                "weight": "DT_FLOAT",
                "attrs": v_attrs
            }],
            "edges":
            edges
        }
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)
        print(f'write {self.name} schema done')

    def convert_to_txt(self):
        edges, id2index = self._load_training_data(
            os.path.join(self.raw_dir, 'train.txt'))
        feature_file = os.path.join(self.raw_dir, 'feature.txt')
        has_feature = os.path.exists(feature_file)
        self.convert_to_schema(len(edges), has_feature)
        if has_feature:
            features, feature_dim = self._load_feature_data(
                os.path.join(self.raw_dir, 'feature.txt'))
            print(f'feature dim is {feature_dim}')
        with open(self.vertex_txt_path, 'w') as f:
            # use index as vertex id
            for k, v in id2index.items():
                if has_feature:
                    features_str = ','.join(str(x) for x in features[k])
                    f.write(f'0\t{v}\t1\t{k}\t{features_str}\n')
                else:
                    f.write(f'0\t{v}\t1\t{k}\n')
        tr_data = dict()
        with open(self.edge_txt_path, 'w') as f:
            for tp, v in edges.items():
                if tp not in tr_data:
                    tr_data[tp] = []
                for e in v:
                    s, d = id2index[e[0]], id2index[e[1]]
                    f.write(f'{tp}\t{s}\t{d}\t1\n')
                    f.write(f'{tp}\t{d}\t{s}\t1\n')
                    tr_data[tp].extend([s, d])
                no_nbrs = set(id2index.values()) - set(tr_data[tp])
                # add self loop
                for vv in no_nbrs:
                    f.write(f'{tp}\t{vv}\t{vv}\t1\n')
                print(f'add self loop number: {len(no_nbrs)} for {tp}')
        tr_nodes = 0
        for k in tr_data.keys():
            tr_data[k] = list(set(tr_data[k]))
            tr_nodes += len(tr_data[k])
        print(f'Total training nodes: {tr_nodes}, '
              f'total nodes: {len(id2index)}, '
              f'edge types: {len(tr_data)}')
        meta = dict(
            edge_type_num=len(tr_data),
            train_num_nodes=tr_nodes,
            max_id=len(id2index) - 1,
        )
        if has_feature:
            meta['feature_dim'] = feature_dim
        with open(os.path.join(self.output_dir, 'train_file'), 'wb') as f:
            pickle.dump(tr_data, f)
        v_data = self._load_testing_data(
            os.path.join(self.raw_dir, 'valid.txt'), id2index)
        with open(os.path.join(self.output_dir, 'eval_file'), 'wb') as f:
            pickle.dump(v_data, f)
        t_data = self._load_testing_data(
            os.path.join(self.raw_dir, 'test.txt'), id2index)
        with open(os.path.join(self.output_dir, 'test_file'), 'wb') as f:
            pickle.dump(t_data, f)
        with open(os.path.join(self.output_dir, 'mata_file'), 'wb') as f:
            pickle.dump(meta, f)

        print(f'convert {self.name} to graph txt files done')

    def get_train_data(self):
        with open(os.path.join(self.output_dir, 'train_file'), 'rb') as f:
            return pickle.load(f)

    def get_eval_data(self):
        with open(os.path.join(self.output_dir, 'eval_file'), 'rb') as f:
            return pickle.load(f)

    def get_test_data(self):
        with open(os.path.join(self.output_dir, 'test_file'), 'rb') as f:
            return pickle.load(f)

    def get_meta_data(self):
        with open(os.path.join(self.output_dir, 'mata_file'), 'rb') as f:
            return pickle.load(f)


def compute_test_metrics(data_source, predict_ouputs):
    predict_dict = dict()
    if isinstance(predict_ouputs, list):
        # list for estimator
        for output in predict_ouputs:
            if output['types'] not in predict_dict:
                predict_dict[output['types']] = dict()
            predict_type = predict_dict[output['types']]
            if output['ids'] not in predict_type:
                predict_type[output['ids']] = list()
            predict_type[output['ids']].append(output['embeddings'])
    elif isinstance(predict_ouputs, dict):
        for tp, ids, embeddings in zip(predict_ouputs['types'],
                                       predict_ouputs['ids'],
                                       predict_ouputs['embeddings']):
            if tp not in predict_dict:
                predict_dict[tp] = dict()
            predict_type = predict_dict[tp]
            if ids not in predict_type:
                predict_type[ids] = list()
            predict_type[ids].append(embeddings)
    else:
        raise ValueError('error type for predict_ouputs')

    true_test, false_test = data_source.get_test_data()
    test_aucs, test_f1s = [], []

    def get_score(tp, v1, v2):
        e1 = np.array(predict_dict[tp][v1]).mean(axis=0)
        e2 = np.array(predict_dict[tp][v2]).mean(axis=0)
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    for i in true_test.keys():
        true_list = list()
        prediction_list = list()
        true_num = 0
        for edge in true_test[i]:
            tmp_score = get_score(i, edge[0], edge[1])
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

        for edge in false_test[i]:
            tmp_score = get_score(i, edge[0], edge[1])
            true_list.append(0)
            prediction_list.append(tmp_score)

        sorted_pred = prediction_list[:]
        sorted_pred.sort()
        threshold = sorted_pred[-true_num]

        y_pred = np.zeros(len(prediction_list), dtype=np.int32)
        for i in range(len(prediction_list)):
            if prediction_list[i] >= threshold:
                y_pred[i] = 1

        y_true = np.array(true_list)
        y_scores = np.array(prediction_list)
        test_aucs.append(roc_auc_score(y_true, y_scores))
        test_f1s.append(f1_score(y_true, y_pred))

    return np.mean(test_aucs), np.mean(test_f1s)


def test():
    pass


if __name__ == "__main__":
    test()
