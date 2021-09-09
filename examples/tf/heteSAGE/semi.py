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
使用Galileo训练heteSAGE模型 semi-supervised
适用user-item异构图，user和item之间存在多种类型的边，包含sparse和dense特征。
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

    def train_data(self):
        assert 'input_file' in self.config
        input_file_num_cols = self.config.get('input_file_num_cols', 3)
        assert input_file_num_cols >= 3

        def base_dataset(**kwargs):
            return gt.TextLineDataset(
                file_pattern='*.csv',
                shuffle=True,
                repeat=True,
                **kwargs,
            )

        def transform(line):
            line = tf.cast(line, tf.int64)
            cols = tf.split(line, input_file_num_cols, axis=-1)
            targets_contexts = tf.nest.map_structure(
                lambda x: tf.reshape(x, [-1]), cols[:-1])
            labels = tf.reshape(cols[-1], [-1, 1])
            return dict(targets=targets_contexts[0],
                        contexts=targets_contexts[1:],
                        labels=labels)

        return gt.dataset_pipeline(base_dataset, transform, **self.config)

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
            labels = tf.random.uniform([10, 1], maxval=2, dtype=tf.int64)
            return gt.TensorDataset(
                dict(
                    targets=targets,
                    contexts=contexts,
                    labels=labels,
                ))

        return gt.dataset_pipeline(base_dataset, None, **self.config)

    def predict_data(self):
        def base_dataset(**kwargs):
            targets = tf.random.uniform([10],
                                        minval=1,
                                        maxval=3,
                                        dtype=tf.int64)
            return gt.TensorDataset(dict(targets=targets))

        return gt.dataset_pipeline(base_dataset, None, **self.config)


class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def get_config(self):
        config = super().get_config()
        config.update(dict(attention_dim=self.attention_dim))
        return config

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.att_weights_s1 = self.add_weight(name='att_weights_s1',
                                              shape=(input_shape[-1],
                                                     self.attention_dim),
                                              initializer='glorot_normal')
        self.att_weights_s2 = self.add_weight(name='att_weights_s2',
                                              shape=(self.attention_dim, 1),
                                              initializer='glorot_normal')
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape [batch size, groups length, dim]
        # output shape [batch size, dim]
        groups_len = inputs.shape[-2]
        att_scores = tf.nn.softmax(
            tf.reshape(
                tf.matmul(tf.tanh(tf.matmul(inputs, self.att_weights_s1)),
                          self.att_weights_s2), [-1, groups_len]))
        att_scores = tf.reshape(att_scores, [-1, 1, groups_len])
        output = tf.matmul(att_scores, inputs)
        output = tf.squeeze(output, axis=1)
        return output


class HeteSAGEEncoder(tf.keras.layers.Layer):
    r'''
    Args:
        num_vertex_types Number of all vertex types
        metapath_groups Groups of metapaths by vertex types,
            4-d list, or dict of 3-d list
        fanouts Fanouts
        embedding_dim Number of embedding dimensions
        feature_dim feature dimension
        aggregator_name
        dropout_rate
        attention_dim Number of attention dimensions

        dense_feature_names list by vertex types
        dense_feature_dims list by vertex types
        sparse_feature_names list by vertex types
        sparse_feature_maxs list by vertex types
        sparse_feature_embedding_dims list by vertex types
    '''
    def __init__(self, num_vertex_types, metapath_groups, fanouts,
                 embedding_dim, feature_dim, aggregator_name, dropout_rate,
                 attention_dim, dense_feature_names, dense_feature_dims,
                 sparse_feature_names, sparse_feature_maxs,
                 sparse_feature_embedding_dims, **kwargs):
        super().__init__()
        assert num_vertex_types == len(metapath_groups)
        assert num_vertex_types == len(dense_feature_names)
        assert num_vertex_types == len(dense_feature_dims)
        assert num_vertex_types == len(sparse_feature_names)
        assert num_vertex_types == len(sparse_feature_maxs)
        assert num_vertex_types == len(sparse_feature_embedding_dims)
        self.num_vertex_types = num_vertex_types
        self.metapath_groups = metapath_groups
        self.fanouts = fanouts
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.aggregator_name = aggregator_name
        self.dropout_rate = dropout_rate
        self.attention_dim = attention_dim
        self.dense_feature_names = dense_feature_names
        self.dense_feature_dims = dense_feature_dims
        self.sparse_feature_names = sparse_feature_names
        self.sparse_feature_maxs = sparse_feature_maxs
        self.sparse_feature_embedding_dims = sparse_feature_embedding_dims

        self.feature_combiners = [
            gt.FeatureCombiner(
                dense_feature_dims=dense_feature_dims[vtype],
                sparse_feature_maxs=sparse_feature_maxs[vtype],
                sparse_feature_embedding_dims=sparse_feature_embedding_dims[
                    vtype],
                hidden_dim=feature_dim,
                feature_combiner='concat') for vtype in range(num_vertex_types)
        ]

        self.num_layers = len(fanouts)
        self.layers = [
            gt.SAGESparseLayer(embedding_dim,
                               aggregator_name,
                               activation='relu',
                               dropout_rate=dropout_rate)
            for _ in range(self.num_layers - 1)
        ]
        self.layers.append(
            gt.SAGESparseLayer(embedding_dim,
                               aggregator_name,
                               dropout_rate=dropout_rate))
        self.relation = gt.RelationTransform(fanouts).transform
        self.fanouts_dim = g.get_fanouts_dim(fanouts)
        self.attention = Attention(attention_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                num_vertex_types=self.num_vertex_types,
                metapath_groups=self.metapath_groups,
                fanouts=self.fanouts,
                embedding_dim=self.embedding_dim,
                feature_dim=self.feature_dim,
                aggregator_name=self.aggregator_name,
                dropout_rate=self.dropout_rate,
                attention_dim=self.attention_dim,
                dense_feature_names=self.dense_feature_names,
                dense_feature_dims=self.dense_feature_dims,
                sparse_feature_maxs=self.sparse_feature_maxs,
                sparse_feature_names=self.sparse_feature_names,
                sparse_feature_embedding_dims=self.
                sparse_feature_embedding_dims,
            ))
        return config

    def call(self, inputs):
        vertices = tf.reshape(inputs, [-1])
        # get types of vertices
        vertices_vtypes = gt.ops.get_pod_feature([vertices], ['vtype'], [1],
                                                 [tf.uint8])[0]
        vertices_vtypes = tf.reshape(vertices_vtypes, [-1])
        features = []
        vtypes_indices = []
        for vtype in range(self.num_vertex_types):
            indices = tf.where(tf.equal(vertices_vtypes, vtype))
            indices = tf.reshape(indices, [-1])
            vtypes_indices.append(indices)
            vertices_vtp = tf.gather(vertices, indices)
            groups_features = []
            # loop all metapath groups for current vertex type
            # aggregate all features for one metapath
            for metapath in self.metapath_groups[vtype]:
                multi_hops = gt.ops.sample_seq_by_multi_hop(
                    vertices=vertices_vtp,
                    metapath=metapath,
                    fanouts=self.fanouts,
                    has_weight=False)[0]
                dup_vertices = tf.reshape(multi_hops, [-1])
                ids, m_indices = tf.unique(dup_vertices)
                feature = self.encode_features(ids)
                relation_graph = self.relation(m_indices)
                for layer in self.layers:
                    relation_graph['feature'] = feature
                    feature = layer(relation_graph)
                output_feature = tf.gather(feature,
                                           relation_graph['target_indices'])
                output_feature = tf.expand_dims(output_feature, axis=1)
                groups_features.append(output_feature)
            groups_features = tf.concat(groups_features, axis=1)
            # merge group features using self-attention
            groups_features = self.attention(groups_features)
            features.append(groups_features)
        features = tf.concat(features, axis=0)
        vtypes_indices = tf.concat(vtypes_indices, axis=0)
        # restore features using inverse indices
        inv_indices = tf.argsort(vtypes_indices)
        features = tf.gather(features, inv_indices)
        return features

    def encode_features(self, vertices):
        # encode features of multi type vertices
        def get_feature(vertices, feature_names, feature_dims, feature_type):
            features = gt.ops.get_pod_feature([vertices], feature_names,
                                              feature_dims, [feature_type] *
                                              len(feature_names))
            features = tf.concat(features, axis=-1)
            return features

        # get types of vertices
        vertices_vtypes = gt.ops.get_pod_feature([vertices], ['vtype'], [1],
                                                 [tf.uint8])[0]
        vertices_vtypes = tf.reshape(vertices_vtypes, [-1])
        feats = []
        all_indices = []
        for vtype in range(self.num_vertex_types):
            indices = tf.where(tf.equal(vertices_vtypes, vtype))
            indices = tf.reshape(indices, [-1])
            all_indices.append(indices)
            vertices_vtp = tf.gather(vertices, indices)
            dense_features = get_feature(vertices_vtp,
                                         self.dense_feature_names[vtype],
                                         self.dense_feature_dims[vtype],
                                         tf.float32)
            sparse_feature_dims = [1] * len(self.sparse_feature_names[vtype])
            sparse_features = get_feature(vertices_vtp,
                                          self.sparse_feature_names[vtype],
                                          sparse_feature_dims, tf.int32)
            feats.append(self.feature_combiners[vtype](
                (dense_features, sparse_features)))
        feats = tf.concat(feats, axis=0)
        indices = tf.concat(all_indices, axis=0)
        # restore features using inverse indices
        inv_indices = tf.argsort(indices)
        feats = tf.gather(feats, inv_indices)
        return feats


class HeteSAGE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop('name'))
        # all args see class HeteSAGEEncoder
        self.metapath_groups = kwargs.pop('metapath_groups')
        assert len(self.metapath_groups) >= 2
        self.encoders = [
            HeteSAGEEncoder(metapath_groups=mg, **kwargs)
            for mg in self.metapath_groups
        ]
        self.attention = Attention(kwargs['attention_dim'])

    def get_config(self):
        config = super().get_config()
        encoder_config = self.encoders[0].get_config()
        encoder_config.update(metapath_groups=self.metapath_groups)
        config.update(encoder_config)
        return config

    def call(self, inputs):
        targets = inputs['targets']
        targets_embedding = self.encoders[0](targets)
        contexts = inputs.get('contexts')
        outputs = dict()
        if contexts is not None:
            labels = inputs['labels']
            labels = tf.cast(labels, tf.float32)
            if len(self.metapath_groups) == 2:
                contexts_embedding = self.encoders[1](contexts)
            else:
                # for multi contexts
                contexts_embeddings = [
                    tf.expand_dims(encoder(c), axis=1)
                    for encoder, c in zip(self.encoders[1:], contexts)
                ]
                contexts_embeddings = tf.concat(contexts_embeddings, axis=1)
                contexts_embedding = self.attention(contexts_embeddings)
            # cosine distance
            normalized_t = tf.nn.l2_normalize(targets_embedding, axis=-1)
            normalized_c = tf.nn.l2_normalize(contexts_embedding, axis=-1)
            cosine = tf.reduce_sum(normalized_t * normalized_c, -1, True)
            logits = tf.nn.sigmoid(cosine)
            outputs['logits'] = logits
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                           logits=logits)
            loss = tf.reduce_mean(loss)
            outputs['loss'] = loss
            self.add_loss(loss)
        else:
            outputs['embeddings'] = targets_embedding
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--model_dir',
                        default='.models/hetesage',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.data_path = '.data/hettest'
    g.start_service_from_args(args)

    batch_size = 64
    batch_num = 100
    num_epochs = 2

    max_id = 100
    num_vertex_types = 2
    # 1-d target,context, 2-d vertex type,
    # 3-d multi groups 4-d metapath 5-d mult edge types
    metapath_groups_0 = [[[0], [1]], [[0], [0]]]
    metapath_groups_1 = [[[1], [0]], [[1], [1]]]
    metapath_groups = [[metapath_groups_0, metapath_groups_1]] * 2
    print(f'metapath groups: {metapath_groups}')
    fanouts = [2, 2]
    embedding_dim = 16
    feature_dim = 8
    attention_dim = 8
    aggregator_name = 'mean'
    dropout_rate = 0.0
    edge_types = [0, 1]
    edge_type_num = len(edge_types)
    sparse_feature_names = [['s1', 's2'], ['s1', 's2', 's3']]
    sparse_feature_maxs = [[5, 5], [5, 5, 5]]
    sparse_feature_embedding_dims = [[8, 8], [8, 8, 8]]
    dense_feature_names = [['d1', 'd2'], ['d1', 'd2']]
    dense_feature_dims = [[2, 2], [1, 2]]

    os.makedirs('/tmp/hetesage', exist_ok=True)
    input_file = '/tmp/hetesage/input_file.csv'
    with open(input_file, 'w') as f:
        f.write('1,2,0\n2,1,1\n')
    input_file = '/tmp/hetesage'

    inputs = Inputs(input_file=input_file,
                    input_file_num_cols=3,
                    batch_size=batch_size,
                    num_workers=1,
                    num_parallel_calls=5)

    model_args = dict(
        num_vertex_types=num_vertex_types,
        metapath_groups=metapath_groups,
        fanouts=fanouts,
        embedding_dim=embedding_dim,
        feature_dim=feature_dim,
        aggregator_name=aggregator_name,
        dropout_rate=dropout_rate,
        attention_dim=attention_dim,
        dense_feature_names=dense_feature_names,
        dense_feature_dims=dense_feature_dims,
        sparse_feature_maxs=sparse_feature_maxs,
        sparse_feature_names=sparse_feature_names,
        sparse_feature_embedding_dims=sparse_feature_embedding_dims,
        name='HeteSAGE',
    )

    is_multi_gpu = len(args.gpu.split(',')) > 1
    trainer = gt.EstimatorTrainer(
        HeteSAGE,
        inputs,
        distribution_strategy='mirrored' if is_multi_gpu else None,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
        model_args=model_args,
    )

    # empty the save_predict_fn, use output of predict
    def custom_save_predict_fn(*arg, **kwargs):
        pass

    def custom_metric_fn(features, predictions):
        labels = features['labels']
        auc = tf.keras.metrics.AUC()
        auc.update_state(y_true=labels, y_pred=predictions['logits'])
        return {'auc': auc}

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
        custom_metric_fn=custom_metric_fn,
    )
    trainer.train(**model_config)
    model_config['batch_size'] = 1024
    outputs = trainer.predict(**model_config)[0]
    print(f'HeteSAGE test output: {len(outputs)}, {outputs[0].keys()}')


if __name__ == "__main__":
    main()
