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
使用Galileo训练GATNE-I模型，使用自定义图数据
构建user-item异构图，user和item之间存在多种类型的边，包含sparse和dense特征。
'''

import os
import argparse
import numpy as np
import functools
import galileo as g
import galileo.tf as gt
import tensorflow as tf


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

    def train_rw_transform(self, vertices, etype):
        walk_length = self.config['walk_length']
        repetition = self.config['repetition']
        context_size = self.config['context_size']
        vertices = tf.cast(tf.reshape(vertices, [-1]), tf.int64)
        pair = gt.ops.sample_pairs_by_random_walk(
            vertices=vertices,
            metapath=[[etype]] * walk_length,
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

    def train_data(self):
        edge_types = self.config['edge_types']
        metapath = self.config['metapath']
        batch_size = self.config.get('batch_size') or 64
        v_type = metapath[0]

        def base_dataset(**kwargs):
            datasets = []
            # one etype one dataset
            for etype in edge_types:
                ds = gt.VertexDataset([v_type], batch_size).map(
                    functools.partial(self.train_rw_transform, etype=etype),
                    num_parallel_calls=5,
                    deterministic=False).unbatch()
                datasets.append(ds)
            ds = datasets[0]
            for i in range(1, len(datasets)):
                ds = ds.concatenate(datasets[i])
            return ds

        ds = gt.dataset_pipeline(base_dataset, None, **self.config)
        return ds

    def evaluate_data(self):
        def base_dataset(**kwargs):
            targets = tf.random.uniform([10],
                                        minval=1,
                                        maxval=3,
                                        dtype=tf.int64)
            contexts = tf.random.uniform([10],
                                         minval=1,
                                         maxval=3,
                                         dtype=tf.int64)
            types = tf.random.uniform([10], maxval=2, dtype=tf.int32)
            return gt.TensorDataset(
                dict(
                    targets=targets,
                    contexts=contexts,
                    types=types,
                ))

        return gt.dataset_pipeline(base_dataset, None, **self.config)

    def predict_data(self):
        def base_dataset(**kwargs):
            targets = tf.random.uniform([10],
                                        minval=1,
                                        maxval=3,
                                        dtype=tf.int64)
            types = tf.random.uniform([10], maxval=2, dtype=tf.int32)
            return gt.TensorDataset(dict(
                targets=targets,
                types=types,
            ))

        return gt.dataset_pipeline(base_dataset, None, **self.config)


class GATNE_I(tf.keras.Model):
    r'''
    Args:
        edge_type_num Number of edge type
        embedding_size Number of embedding dimensions
        edge_embedding_size Number of edge embedding dimensions
        attention_dim Number of attention dimensions
        num_vertexes number of vertex
        negative_num Negative samples for optimization
        neighbor_samples Number of neighbor samples
        feature_dim feature dimension
        vertex_type
    '''
    def __init__(self, edge_type_num, embedding_size, edge_embedding_size,
                 attention_dim, num_vertexes, negative_num, neighbor_samples,
                 feature_dim, vertex_type, edge_types, dense_feature_names,
                 dense_feature_dims, sparse_feature_maxs, sparse_feature_names,
                 sparse_feature_embedding_dims, **kwargs):
        super().__init__()
        self.edge_type_num = edge_type_num
        self.embedding_size = embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.attention_dim = attention_dim
        self.num_vertexes = num_vertexes
        self.negative_num = negative_num
        self.neighbor_samples = neighbor_samples
        self.feature_dim = feature_dim
        self.vertex_type = vertex_type
        self.edge_types = edge_types
        assert len(sparse_feature_names) == len(dense_feature_names)
        assert len(sparse_feature_maxs) == len(dense_feature_names)
        assert len(sparse_feature_embedding_dims) == len(dense_feature_names)
        assert len(dense_feature_dims) == len(dense_feature_names)
        assert len(vertex_type) == len(dense_feature_names)
        self.dense_feature_names = dense_feature_names
        self.dense_feature_dims = dense_feature_dims
        self.sparse_feature_names = sparse_feature_names
        self.sparse_feature_maxs = sparse_feature_maxs
        self.sparse_feature_embedding_dims = sparse_feature_embedding_dims

    def build(self, input_shape):
        self.feature_combiners = {
            vtype: gt.FeatureCombiner(
                dense_feature_dims=self.dense_feature_dims[vtype],
                sparse_feature_maxs=self.sparse_feature_maxs[vtype],
                sparse_feature_embedding_dims=self.
                sparse_feature_embedding_dims[vtype],
                hidden_dim=self.feature_dim,
                feature_combiner='concat')
            for vtype in self.vertex_type
        }

        # transform weights
        self.feature_trans = self.add_weight(
            name='feature_trans',
            shape=(self.feature_dim, self.embedding_size),
            initializer=tf.initializers.RandomNormal(0.0, 1.0))
        # need this node_trans?
        self.node_trans = self.add_weight(name='node_trans',
                                          shape=(self.feature_dim,
                                                 self.embedding_size),
                                          initializer='glorot_normal')
        self.edge_embedding_trans = self.add_weight(
            name='edge_embedding_trans',
            shape=(self.edge_type_num, self.feature_dim,
                   self.edge_embedding_size),
            initializer='glorot_normal')
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

    def get_features(self, vertices, vtypes):
        # get features of vertices for each vtypes

        # get types of vertices
        vertices_vtypes = gt.ops.get_pod_feature([vertices], ['vtype'], [1],
                                                 [tf.uint8])[0]
        vertices_vtypes = tf.reshape(vertices_vtypes, [-1])
        features = dict()
        output_indices = []
        for vtype in vtypes:
            indices = tf.where(tf.equal(vertices_vtypes, vtype))
            vertices_vtp = tf.gather(vertices, indices)
            vertices_vtp = tf.reshape(vertices_vtp, [-1])
            dense_features = gt.ops.get_pod_feature(
                [vertices_vtp], self.dense_feature_names[vtype],
                self.dense_feature_dims[vtype],
                [tf.float32] * len(self.dense_feature_names[vtype]))
            sparse_len = len(self.sparse_feature_names[vtype])
            sparse_features = gt.ops.get_pod_feature(
                [vertices_vtp], self.sparse_feature_names[vtype],
                [1] * sparse_len, [tf.int32] * sparse_len)
            features[vtype] = (dense_features, sparse_features)
            output_indices.append(indices)

        output_indices = tf.concat(output_indices, axis=0)
        return features, output_indices

    def nbr_feature_transform(self, inputs):
        targets, types = inputs['targets'], inputs['types']
        targets = tf.reshape(tf.cast(targets, dtype=tf.int64), [-1])
        types = tf.reshape(types, [-1])

        targets_features, indices = self.get_features(targets,
                                                      self.vertex_type)
        new_targets = tf.reshape(tf.gather(targets, indices), [-1])
        new_types = tf.reshape(tf.gather(types, indices), [-1])

        nbr_features = []
        for nbr_etype in self.edge_types:
            neighbor = gt.ops.sample_neighbors(new_targets, [nbr_etype],
                                               count=self.neighbor_samples,
                                               has_weight=False)[0]
            neighbor = tf.reshape(neighbor, [-1])
            nbr_features.append(
                self.get_features(neighbor, self.vertex_type)[0])
        r'''
        targets shape [N, ]
        types shape [N, ]
        targets_features dict of (dense, sparse) by vtype
        nbr_features list of dict of (dense, sparse), by etype by vtype
        '''
        outputs = dict(
            targets=new_targets,
            types=new_types,
            targets_features=targets_features,
            nbr_features=nbr_features,
        )
        contexts = inputs.get('contexts')
        if contexts is not None:
            outputs['contexts'] = tf.cast(contexts, dtype=tf.int64)
        return outputs

    def call(self, inputs):
        inputs = self.nbr_feature_transform(inputs)
        targets = tf.squeeze(inputs['targets'])
        types = tf.squeeze(inputs['types'])
        targets_features = inputs['targets_features']
        nbr_features = inputs['nbr_features']

        targets_features_tmp = []
        for vtype in self.vertex_type:
            if vtype in targets_features:
                feat = self.feature_combiners[vtype](targets_features[vtype])
                targets_features_tmp.append(feat)
        targets_features = tf.concat(targets_features_tmp, axis=0)
        # [N, feature_dim]

        nbr_features_list = []
        for nbr_features_etype in nbr_features:
            nbr_features_tmp = []
            for vtype in self.vertex_type:
                if vtype in nbr_features_etype:
                    feat = self.feature_combiners[vtype](
                        nbr_features_etype[vtype])
                    nbr_features_tmp.append(feat)
            feats = tf.concat(nbr_features_tmp, axis=0)
            feats = tf.reshape(feats,
                               [-1, self.neighbor_samples, self.feature_dim])
            nbr_features_list.append(tf.expand_dims(feats, axis=1))
        nbr_features = tf.concat(nbr_features_list, axis=1)
        # [N, edge_type_num, neighbor_samples, feature_dim]

        edge_embed_typed = [
            tf.matmul(
                tf.reshape(
                    tf.slice(nbr_features, [0, i, 0, 0], [-1, 1, -1, -1]),
                    [-1, self.feature_dim]),
                tf.reshape(
                    tf.slice(self.edge_embedding_trans, [i, 0, 0],
                             [1, -1, -1]),
                    [self.feature_dim, self.edge_embedding_size]))
            for i in range(self.edge_type_num)
        ]
        edge_embed_tmp = tf.concat(edge_embed_typed, axis=0)
        edge_embed_tmp = tf.reshape(edge_embed_tmp, [
            self.edge_type_num, -1, self.neighbor_samples,
            self.edge_embedding_size
        ])
        edge_agg_embed = tf.transpose(tf.reduce_mean(edge_embed_tmp, axis=2),
                                      perm=[1, 0, 2])

        # attention
        trans_w = tf.nn.embedding_lookup(self.trans_weights, types)
        trans_w_s1 = tf.nn.embedding_lookup(self.trans_weights_s1, types)
        trans_w_s2 = tf.nn.embedding_lookup(self.trans_weights_s2, types)

        attention = tf.reshape(
            tf.nn.softmax(
                tf.reshape(
                    tf.matmul(tf.tanh(tf.matmul(edge_agg_embed, trans_w_s1)),
                              trans_w_s2), [-1, self.edge_type_num])),
            [-1, 1, self.edge_type_num])
        edge_att_embed = tf.matmul(attention, edge_agg_embed)
        edge_att_embed = tf.reshape(tf.matmul(edge_att_embed, trans_w),
                                    [-1, self.embedding_size])

        node_embed = tf.matmul(targets_features, self.node_trans)
        feature_embed = tf.matmul(targets_features, self.feature_trans)
        all_node_embed = node_embed + edge_att_embed + feature_embed
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
                feature_dim=self.feature_dim,
                vertex_type=self.vertex_type,
                edge_types=self.edge_types,
                dense_feature_names=self.dense_feature_names,
                dense_feature_dims=self.dense_feature_dims,
                sparse_feature_maxs=self.sparse_feature_maxs,
                sparse_feature_names=self.sparse_feature_names,
                sparse_feature_embedding_dims=self.
                sparse_feature_embedding_dims,
            ))
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--model_dir',
                        default='.models/gatne-i',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.data_path = '.data/hettest'
    g.start_service_from_args(args)

    batch_size = 64
    num_epochs = 1
    repetition = 20
    neighbor_samples = 10
    walk_length = 10
    context_size = 2

    max_id = 100
    train_num_nodes = max_id + 1
    vertex_type = [0, 1]
    edge_types = [0, 1]
    edge_type_num = len(edge_types)
    metapath = [0, 1, 0]
    feature_dim = 8
    sparse_feature_names = [['s1', 's2'], ['s1', 's2', 's3']]
    sparse_feature_maxs = [[5, 5], [5, 5, 5]]
    sparse_feature_embedding_dims = [[8, 8], [8, 8, 8]]
    dense_feature_names = [['d1', 'd2'], ['d1', 'd2']]
    dense_feature_dims = [[2, 2], [1, 2]]

    # batch_num = train_num_nodes * number pairs of random walk / batch_size
    batch_num = (train_num_nodes * repetition * context_size *
                 (2 * walk_length - context_size + 1) + batch_size -
                 1) // batch_size
    print(f'batch num {batch_num}')

    inputs = Inputs(batch_size=batch_size,
                    num_epochs=num_epochs,
                    vertex_type=vertex_type,
                    edge_types=edge_types,
                    walk_length=walk_length,
                    repetition=repetition,
                    context_size=context_size,
                    neighbor_samples=neighbor_samples,
                    metapath=metapath,
                    num_parallel_calls=5)

    model_args = dict(
        edge_type_num=edge_type_num,
        embedding_size=200,
        edge_embedding_size=10,
        attention_dim=20,
        num_vertexes=max_id + 1,
        negative_num=5,
        neighbor_samples=neighbor_samples,
        feature_dim=feature_dim,
        vertex_type=vertex_type,
        edge_types=edge_types,
        dense_feature_names=dense_feature_names,
        dense_feature_dims=dense_feature_dims,
        sparse_feature_maxs=sparse_feature_maxs,
        sparse_feature_names=sparse_feature_names,
        sparse_feature_embedding_dims=sparse_feature_embedding_dims,
        name='GATNE-I',
    )

    is_multi_gpu = len(args.gpu.split(',')) > 1
    trainer = gt.EstimatorTrainer(
        GATNE_I,
        inputs,
        distribution_strategy='mirrored' if is_multi_gpu else None,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
        model_args=model_args,
    )

    # empty the save_predict_fn, use output of predict
    def custom_save_predict_fn(*arg, **kwargs):
        pass

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
    )
    trainer.train(**model_config)
    model_config['batch_size'] = 1024
    outputs = trainer.predict(**model_config)[0]
    print(f'GATNE-I test output: {len(outputs)}, {outputs[0].keys()}')


if __name__ == "__main__":
    main()
