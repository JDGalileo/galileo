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
使用Galileo训练GATNE-T模型
'''

import os
import argparse
import numpy as np
import functools
import galileo as g
import galileo.tf as gt
import tensorflow as tf
import utils


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

    def train_rw_transform(self, inputs, etype):
        walk_length = self.config['walk_length']
        repetition = self.config['repetition']
        context_size = self.config['context_size']
        vertices = inputs['targets']
        vertices = tf.cast(tf.reshape(vertices, [-1]), tf.int64)
        metapath = [[etype]] * walk_length
        pair = gt.ops.sample_pairs_by_random_walk(
            vertices=vertices,
            metapath=metapath,
            repetition=repetition,
            context_size=context_size,
            p=1,
            q=1,
        )
        targets, contexts = tf.split(pair, [1, 1], axis=-1)
        targets = tf.reshape(targets, [-1])
        contexts = tf.reshape(contexts, [-1])
        types = tf.repeat(tf.constant([etype], dtype=tf.int64),
                          tf.size(targets))
        # targets shape [N, ]
        # contexts shape [N, ]
        # types shape [N, ]
        return dict(targets=targets, contexts=contexts, types=types)

    def nbr_transform(self, inputs):
        edge_types = self.config['edge_types']
        neighbor_samples = self.config['neighbor_samples']

        targets, types = inputs['targets'], inputs['types']
        targets = tf.reshape(tf.cast(targets, dtype=tf.int64), [-1])
        types = tf.reshape(types, [-1])
        neighbors_tmp = []
        for nbr_etype in edge_types:
            neighbor = gt.ops.sample_neighbors(targets, [nbr_etype],
                                               count=neighbor_samples,
                                               has_weight=False)[0]
            neighbors_tmp.append(tf.expand_dims(neighbor, axis=1))
        neighbors = tf.concat(neighbors_tmp, axis=1)
        # targets shape [N, ]
        # types shape [N, ]
        # neighbors shape [N, edge_type_num, neighbor_samples)]
        outputs = dict(targets=targets, types=types, neighbors=neighbors)
        contexts = inputs.get('contexts')
        if contexts is not None:
            outputs['contexts'] = tf.cast(contexts, dtype=tf.int64)
        return outputs

    def train_data(self):
        data_source = self.config['data_source']
        batch_size = self.config.get('batch_size') or 64
        num_epochs = self.config.get('num_epochs') or 10

        def base_dataset(**kwargs):
            data = data_source.get_train_data()
            assert len(data) > 0
            # one etype one dataset
            datasets = []
            for etype, d in data.items():
                ds = gt.TensorDataset(dict(targets=d)).shuffle(
                    len(d),
                    reshuffle_each_iteration=True).batch(batch_size).map(
                        functools.partial(self.train_rw_transform,
                                          etype=etype),
                        num_parallel_calls=5,
                        deterministic=False).unbatch()
                datasets.append(ds)
            ds = datasets[0]
            for i in range(1, len(datasets)):
                ds = ds.concatenate(datasets[i])
            ds = ds.repeat(num_epochs)
            return ds

        ds = gt.dataset_pipeline(base_dataset, self.nbr_transform,
                                 **self.config)
        return ds

    def evaluate_data(self):
        data_source = self.config['data_source']

        def base_dataset(**kwargs):
            true_eval, false_eval = data_source.get_eval_data()
            targets = []
            contexts = []
            types = []
            for tp, v in true_eval.items():
                edge = np.array(v)
                targets.extend(edge[:, 0].tolist())
                contexts.extend(edge[:, 1].tolist())
                types.extend([tp] * len(v))
            for tp, v in false_eval.items():
                edge = np.array(v)
                targets.extend(edge[:, 0].tolist())
                contexts.extend(edge[:, 1].tolist())
                types.extend([tp] * len(v))
            print(f'All evaluate edges is {len(targets)}')
            return gt.TensorDataset(
                dict(
                    targets=targets,
                    contexts=contexts,
                    types=types,
                ))

        return gt.dataset_pipeline(base_dataset, self.nbr_transform,
                                   **self.config)

    def predict_data(self):
        data_source = self.config['data_source']

        def base_dataset(**kwargs):

            true_test, false_test = data_source.get_test_data()
            vertexes = []
            types = []
            for tp, v in true_test.items():
                vert = np.unique(np.array(v))
                vertexes.extend(vert.tolist())
                types.extend([tp] * len(vert))
            for tp, v in false_test.items():
                vert = np.unique(np.array(v))
                vertexes.extend(vert.tolist())
                types.extend([tp] * len(vert))
            print(f'All test vertexes is {len(vertexes)}')
            return gt.TensorDataset(dict(
                targets=vertexes,
                types=types,
            ))

        return gt.dataset_pipeline(base_dataset, self.nbr_transform,
                                   **self.config)


class GATNE_T(tf.keras.Model):
    r'''
    Args:
        edge_type_num Number of edge type
        embedding_size Number of embedding dimensions
        edge_embedding_size Number of edge embedding dimensions
        attention_dim Number of attention dimensions
        num_vertexes number of vertex
        negative_num Negative samples for optimization
    '''
    def __init__(self, edge_type_num, embedding_size, edge_embedding_size,
                 attention_dim, num_vertexes, negative_num, neighbor_samples,
                 **kwargs):
        super().__init__()
        self.edge_type_num = edge_type_num
        self.embedding_size = embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.attention_dim = attention_dim
        self.num_vertexes = num_vertexes
        self.negative_num = negative_num
        self.neighbor_samples = neighbor_samples

    def build(self, input_shape):
        #bs = input_shape['targets'][0]
        # embeddings
        self.base_node_embeddings = self.add_weight(
            name='base_node_embeddings',
            shape=(self.num_vertexes, self.embedding_size),
            initializer=tf.initializers.RandomUniform(-1.0, 1.0))
        self.node_type_embeddings = self.add_weight(
            name='node_type_embeddings',
            shape=(self.num_vertexes, self.edge_type_num,
                   self.edge_embedding_size),
            initializer=tf.initializers.RandomUniform(-1.0, 1.0))

        # transform weights
        self.trans_weights = self.add_weight(name='trans_weights',
                                             shape=(self.edge_type_num,
                                                    self.edge_embedding_size,
                                                    self.embedding_size),
                                             initializer='glorot_normal')

        self.trans_weights_s1 = self.add_weight(
            name='trans_weights_s1',
            shape=(self.edge_type_num, self.edge_embedding_size,
                   self.attention_dim),
            initializer='glorot_normal')
        self.trans_weights_s2 = self.add_weight(name='trans_weights_s2',
                                                shape=(self.edge_type_num,
                                                       self.attention_dim, 1),
                                                initializer='glorot_normal')

        # nce weights
        self.nce_weights = self.add_weight(name='nce_weights',
                                           shape=(self.num_vertexes,
                                                  self.embedding_size),
                                           initializer='glorot_normal')

        self.nce_biases = self.add_weight(name='nce_biases',
                                          shape=(self.num_vertexes, ),
                                          initializer='zeros')

    def call(self, inputs):
        targets = tf.squeeze(inputs['targets'])
        types = tf.squeeze(inputs['types'])
        neighbors = inputs['neighbors']

        # lookup neighbors embeddings and mean them
        node_embed_neighbors = tf.nn.embedding_lookup(
            self.node_type_embeddings, neighbors)

        node_embed_typed = [
            tf.reshape(
                tf.slice(node_embed_neighbors, [0, i, 0, i, 0],
                         [-1, 1, -1, 1, -1]),
                [1, -1, self.neighbor_samples, self.edge_embedding_size])
            for i in range(self.edge_type_num)
        ]
        node_embed_tmp = tf.concat(node_embed_typed, axis=0)
        node_agg_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2),
                                      perm=[1, 0, 2])

        # attention
        trans_w = tf.nn.embedding_lookup(self.trans_weights, types)
        trans_w_s1 = tf.nn.embedding_lookup(self.trans_weights_s1, types)
        trans_w_s2 = tf.nn.embedding_lookup(self.trans_weights_s2, types)

        attention = tf.reshape(
            tf.nn.softmax(
                tf.reshape(
                    tf.matmul(tf.tanh(tf.matmul(node_agg_embed, trans_w_s1)),
                              trans_w_s2), [-1, self.edge_type_num])),
            [-1, 1, self.edge_type_num])
        node_att_embed = tf.matmul(attention, node_agg_embed)

        node_embed = tf.nn.embedding_lookup(self.base_node_embeddings, targets)
        all_node_embed = node_embed + tf.reshape(
            tf.matmul(node_att_embed, trans_w), [-1, self.embedding_size])
        embeddings = tf.nn.l2_normalize(all_node_embed, axis=1)

        contexts = inputs.get('contexts')
        outputs = dict()
        if contexts is not None:
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=tf.reshape(contexts, [-1, 1]),
                               inputs=embeddings,
                               num_sampled=self.negative_num,
                               num_classes=self.num_vertexes))
            outputs['loss'] = loss
            self.add_loss(loss)
        else:
            outputs['ids'] = targets
            outputs['types'] = types
            outputs['embeddings'] = embeddings
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                edge_type_num=self.edge_type_num,
                embedding_size=self.embedding_size,
                edge_embedding_size=self.edge_embedding_size,
                attention_dim=self.attention_dim,
                num_vertexes=self.num_vertexes,
                negative_num=self.negative_num,
                neighbor_samples=self.neighbor_samples,
            ))
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-source',
        '-i',
        default='amazon',
        type=str,
        help='data source name, amazon example twitter youtube')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--model_dir',
                        default='.models/gatne-t',
                        type=str,
                        help='model dir')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_source = utils.DataSource(args.data_source)
    args.data_path = data_source.output_dir
    g.start_service_from_args(args)

    batch_size = 64
    num_epochs = args.epochs
    repetition = 20
    neighbor_samples = 10
    walk_length = 10
    context_size = 2

    meta = data_source.get_meta_data()
    max_id = meta['max_id']  # 10098 for amazon
    train_num_nodes = meta['train_num_nodes']  # 13185 for amazon
    edge_type_num = meta['edge_type_num']
    # batch_num = train_num_nodes * number pairs of random walk / batch_size
    # 156572
    batch_num = (train_num_nodes * repetition * context_size *
                 (2 * walk_length - context_size + 1) + batch_size -
                 1) // batch_size
    print(f'Data Source {args.data_source}, max id {max_id}, '
          f'number of train nodes {train_num_nodes}, batch num {batch_num}')

    if args.debug:
        batch_num = 200

    inputs = Inputs(data_source=data_source,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    vertex_type=[0],
                    edge_types=list(range(edge_type_num)),
                    walk_length=walk_length,
                    repetition=repetition,
                    context_size=context_size,
                    neighbor_samples=neighbor_samples,
                    num_parallel_calls=5)

    model_args = dict(
        edge_type_num=edge_type_num,
        embedding_size=200,
        edge_embedding_size=10,
        attention_dim=20,
        num_vertexes=max_id + 1,
        negative_num=5,
        neighbor_samples=neighbor_samples,
        name='GATNE-T',
    )

    is_multi_gpu = len(args.gpu.split(',')) > 1
    trainer = gt.EstimatorTrainer(
        GATNE_T,
        inputs,
        distribution_strategy='mirrored' if is_multi_gpu else None,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
        model_args=model_args,
    )

    # empty the save_predict_fn, use output of predict
    def custom_save_predict_fn(*arg, **kwargs):
        pass

    def early_stop_hook(estimator, **kwargs):
        return [
            tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator,
                'loss',
                max_steps_without_decrease=batch_num * 5,
                run_every_secs=300,
                run_every_steps=None)
        ]

    model_config = dict(
        batch_size=batch_size,
        batch_num=batch_num,
        num_epochs=num_epochs,
        max_id=max_id,
        model_dir=args.model_dir,
        save_checkpoint_epochs=1,
        log_steps=1000,
        optimizer='adam',
        learning_rate=0.001,
        train_verbose=2,
        save_predict_fn=custom_save_predict_fn,
        eval_exporters=gt.BestCheckpointsExporter(max_to_keep=3),
        eval_throttle_secs=600,
        estimator_hooks_fn=early_stop_hook,
    )
    trainer.train(**model_config)
    model_config['batch_size'] = 1024
    outputs = trainer.predict(**model_config)
    tm = utils.compute_test_metrics(data_source, outputs[0])
    print(f'Test auc: {tm[0]:.6f}, f1: {tm[1]:.6f}')


if __name__ == "__main__":
    main()
