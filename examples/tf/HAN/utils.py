import os
import json
import pickle
import scipy.io as scio
import numpy as np
import galileo as g
import galileo.tf as gt
import tensorflow as tf
from sklearn.metrics import f1_score


class DataSource(g.DataSource):
    # url = 'https://github.com/Jhy1993/HAN/tree/master/data/acm'
    url = 'http://storage.jd.local/pinoctl/graphdata/han'

    def __init__(self, name, root_dir='./.data', **kwargs):
        assert name.lower() in ('acm')
        # download and convert in init
        super().__init__(root_dir, name, **kwargs)

    @property
    def raw_file_names(self):
        names = ['ACM.mat']
        return names

    def download(self):
        for name in self.raw_file_names:
            g.download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)
            acm_data = scio.loadmat(os.path.join(self.raw_dir, name))
        print(f'download {self.name} done')

        p_vs_l, p_vs_a, p_vs_t, p_vs_c = acm_data['PvsL'], acm_data[
            'PvsA'], acm_data['PvsT'], acm_data['PvsC']
        label_ids = [0, 1, 2, 2, 1]
        conf_ids = [0, 1, 9, 10, 13]
        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        self.p_vs_l, self.p_vs_a, p_vs_t, p_vs_c = p_vs_l[p_selected], p_vs_a[
            p_selected], p_vs_t[p_selected], p_vs_c[p_selected]
        self.features = p_vs_t.toarray()
        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        self.labels = labels

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(
                np.linspace(0, 1, pc_c_mask.sum()))
        self.train_idx = np.where(float_mask <= 0.7)[0]
        self.val_idx = np.where((float_mask > 0.7) & (float_mask <= 0.9))[0]
        self.test_idx = np.where(float_mask > 0.9)[0]

    def convert_to_schema(self, edge_type_num=None, has_feature=False):

        self.feature_dim = self.features.shape[1]
        schema = {'vertexes': [], 'edges': []}
        schema['vertexes'] = [{}, {}, {}]
        schema['edges'] = [{}, {}, {}, {}]
        for i in range(3):
            schema['vertexes'][i] = {
                'vtype': i,
                'entity': 'DT_INT64',
                "weight": "DT_FLOAT",
                "attrs": []
            }
        schema['vertexes'][0]['attrs']=[{'name':'feature', "dtype":"DT_ARRAY_FLOAT"}, \
            {'name':'label', "dtype":"DT_INT64"}]

        for i in range(4):
            schema['edges'][i] = {
                "etype": i,
                "entity_1": "DT_INT64",
                "entity_2": "DT_INT64",
                "weight": "DT_FLOAT",
                "attrs": []
            }

        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)
        print(f'write {self.name} schema done')

    def convert_to_txt(self):
        with open(self.vertex_txt_path, 'w') as f:
            paper_num, author_num, filed_num = self.p_vs_a.shape[
                0], self.p_vs_a.shape[1], self.p_vs_l.shape[1]
            for i in range(paper_num):
                features_str = ','.join(str(x) for x in self.features[i])
                label_str = str(self.labels[i])
                f.write(f'0\t{str(i)}\t1\t{features_str}\t{label_str}\n')
            for i in range(author_num):
                f.write(f'1\t{str(i+paper_num)}\t1\n')
            for i in range(filed_num):
                f.write(f'2\t{str(i+author_num+paper_num)}\t1\n')

        with open(self.edge_txt_path, 'w') as f:
            p, a = self.p_vs_a.nonzero()

            for i in range(len(p)):
                f.write(f'0\t{str(p[i])}\t{str(a[i]+paper_num)}\t1\n')
                f.write(f'1\t{str(a[i]+paper_num)}\t{str(p[i])}\t1\n')

            set_p, set_a = set(p), set(a)
            non_p_vs_a = [i for i in range(paper_num) if i not in set_p]
            non_a_vs_p = [i for i in range(author_num) if i not in set_a]
            for i in range(len(non_p_vs_a)):
                f.write(f'0\t{str(non_p_vs_a[i])}\t{str(non_p_vs_a[i])}\t1\n')
                f.write(f'1\t{str(non_p_vs_a[i])}\t{str(non_p_vs_a[i])}\t1\n')
            for i in range(len(non_a_vs_p)):
                f.write(
                    f'1\t{str(non_a_vs_p[i]+paper_num)}\t{str(non_a_vs_p[i]+paper_num)}\t1\n'
                )

            p, l = self.p_vs_l.nonzero()
            for i in range(len(p)):
                f.write(
                    f'2\t{str(p[i])}\t{str(l[i]+paper_num+author_num)}\t1\n')
                f.write(
                    f'3\t{str(l[i]+paper_num+author_num)}\t{str(p[i])}\t1\n')

            set_p, set_l = set(p), set(l)
            non_p_vs_l = [i for i in range(paper_num) if i not in set_p]
            non_l_vs_p = [i for i in range(filed_num) if i not in set_l]
            for i in range(len(non_p_vs_l)):
                f.write(f'2\t{str(non_p_vs_l[i])}\t{str(non_p_vs_l[i])}\t1\n')
                f.write(f'3\t{str(non_p_vs_l[i])}\t{str(non_p_vs_l[i])}\t1\n')
            for i in range(len(non_l_vs_p)):
                f.write(
                    f'3\t{str(non_l_vs_p[i]+paper_num+author_num)}\t{str(non_l_vs_p[i]+paper_num+author_num)}\t1\n'
                )

        print(f'edge_len:{2*(len(a)+len(l))}')
        print(f'convert {self.name} to graph txt files done')

        with open(os.path.join(self.output_dir, 'train_file'), 'wb') as f:
            pickle.dump(self.train_idx, f)
        with open(os.path.join(self.output_dir, 'eval_file'), 'wb') as f:
            pickle.dump(self.val_idx, f)
        with open(os.path.join(self.output_dir, 'test_file'), 'wb') as f:
            pickle.dump(self.test_idx, f)
        with open(os.path.join(self.output_dir, 'test_label_file'), 'wb') as f:
            pickle.dump(self.labels[self.test_idx], f)
        print(
            f'train_idx len:{len(self.train_idx)}\tval_idx len:{len(self.val_idx)}\t test_idx len:{len(self.test_idx)}'
        )

    def get_train_data(self):
        with open(os.path.join(self.output_dir, 'train_file'), 'rb') as f:
            return pickle.load(f)

    def get_eval_data(self):
        with open(os.path.join(self.output_dir, 'eval_file'), 'rb') as f:
            return pickle.load(f)

    def get_test_data(self):
        with open(os.path.join(self.output_dir, 'test_file'), 'rb') as f:
            return pickle.load(f)

    def get_test_labels(self):
        with open(os.path.join(self.output_dir, 'test_file'), 'rb') as f:
            idx = pickle.load(f)
        with open(os.path.join(self.output_dir, 'test_label_file'), 'rb') as f:
            labels = pickle.load(f)
        return {idx[i]: labels[i] for i in range(len(idx))}


def compute_test_metrics(data_source, predict_ouputs):
    logits_dict = {}

    for output in predict_ouputs:
        if output['ids'] not in logits_dict: logits_dict[output['ids']] = []
        logits_dict[output['ids']].append(output['logits'])

    labels_dict = data_source.get_test_labels()
    labels, logits = [], []

    for ids in labels_dict:
        if ids in logits_dict:
            labels.append(labels_dict[ids])
            logits.append(logits_dict[ids])
    labels, logits = np.array(labels), np.array(logits)
    logits = np.argmax(logits, axis=-1)

    micro_f1 = f1_score(labels, logits, average='micro')
    macro_f1 = f1_score(labels, logits, average='macro')
    return {'micro_f1': micro_f1, 'macro_f1': macro_f1}
