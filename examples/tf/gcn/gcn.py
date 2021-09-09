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
使用Galileo训练GCN模型
'''

import os
import argparse
import numpy as np
import functools
import galileo as g
import galileo.tf as gt
import tensorflow as tf


class GCN(tf.keras.Model):
    def __init__(self,
                 edge_types: list,
                 max_id: int,
                 feature_name: str,
                 feature_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 2,
                 bias: bool = True,
                 dropout_rate: float = 0.0,
                 normalization=None,
                 **kwargs):
        super().__init__(name='GCN')
        self.edge_types = edge_types
        self.max_id = max_id
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.normalization = normalization

        self._layers = [
            gt.GCNLayer(hidden_dim,
                        bias=bias,
                        dropout_rate=dropout_rate,
                        activation='relu',
                        normalization=normalization)
            for _ in range(self.num_layers - 1)
        ]
        self._layers.append(
            gt.GCNLayer(num_classes,
                        bias=bias,
                        dropout_rate=0.0,
                        normalization=normalization))

    def call(self, inputs):
        graph = self._get_graph_data(inputs)
        targets = inputs['targets']
        labels = inputs['labels']
        for layer in self._layers:
            features = layer(graph)
            graph['features'] = features
        logits = tf.gather(features, targets)
        # compute loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=labels)
        loss = tf.reduce_mean(losses)
        self.add_loss(loss)
        return dict(loss=loss, logits=logits)

    def _get_graph_data(self, inputs):
        vertices = inputs['targets']
        full_nbrs = gt.ops.get_full_neighbors(vertices,
                                              self.edge_types,
                                              has_weight=True)
        # full_nbrs already have self loop
        edge_dsts, edge_weights, idx = full_nbrs
        # second col of idx is degree
        degs = tf.split(idx, 2, axis=1)[1]
        degs = tf.reshape(degs, [-1])
        edge_srcs = tf.repeat(vertices, degs)
        all_vertices = tf.range(self.max_id + 1, dtype=tf.int64)
        features = gt.ops.get_pod_feature([all_vertices], [self.feature_name],
                                          [self.feature_dim], [tf.float32])[0]
        return dict(
            vertices=vertices,  # [num vertices]
            edge_srcs=edge_srcs,  # [num edges]
            edge_dsts=edge_dsts,  # [num edges]
            edge_weights=edge_weights,  # [num edges]
            features=features,  #[all vertices, feature dim]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                edge_types=self.edge_types,
                max_id=self.max_id,
                feature_name=self.feature_name,
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout_rate=self.dropout_rate,
                normalization=self.normalization,
            ))
        return config


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

    def transform(self, vertices):
        label_name = self.config['label_name']
        label_dim = self.config['label_dim']
        vertices = tf.cast(vertices, tf.int64)
        vertices = tf.reshape(vertices, [-1])
        u_vertices, _ = tf.unique(vertices)
        labels = gt.ops.get_pod_feature([u_vertices], [label_name],
                                        [label_dim], [tf.float32])[0]
        return dict(targets=u_vertices, labels=labels)

    def train_data(self):
        vertex_type = self.config['vertex_type']

        def base_dataset(**kwargs):
            # make sure sample all train vertices (1208 for cora)
            return gt.VertexDataset(vertex_type, 10000)

        return gt.dataset_pipeline(base_dataset, self.transform, **self.config)

    def evaluate_data(self):
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gt.dataset_pipeline(
            lambda **kwargs: gt.TensorDataset(test_ids, **kwargs),
            self.transform, **self.config)

    def predict_data(self):
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gt.dataset_pipeline(
            lambda **kwargs: gt.TensorDataset(test_ids, **kwargs),
            self.transform, **self.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_id', default=2707, type=int, help='max node id')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--ds',
                        default=None,
                        type=str,
                        help='distribution strategy '
                        '(mirrored, multi_worker_mirrored, parameter_server)')
    parser.add_argument('--feature_dim',
                        default=1433,
                        type=int,
                        help='dense feature dimemsion')
    parser.add_argument('--label_dim',
                        default=7,
                        type=int,
                        help='label feature dimemsion')
    parser.add_argument('--model_dir',
                        default='.models/gcn_tf',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inputs = Inputs(vertex_type=[0],
                    label_name='label',
                    label_dim=args.label_dim,
                    data_source_name=args.data_source_name)

    model_args = dict(
        edge_types=[0],
        max_id=args.max_id,
        feature_name='feature',
        feature_dim=args.feature_dim,
        hidden_dim=64,
        num_classes=args.label_dim,
        num_layers=2,
        dropout_rate=0.0,
    )

    trainer = gt.EstimatorTrainer(
        GCN,
        inputs,
        model_args=model_args,
        distribution_strategy=args.ds,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
    )

    def custom_metric_fn(features, predictions):
        labels = features['labels']
        acc = tf.keras.metrics.CategoricalAccuracy()
        acc.update_state(y_true=labels, y_pred=predictions['logits'])
        return {'acc': acc}

    model_config = dict(
        batch_size=64,
        batch_num=1,
        num_epochs=20,
        max_id=args.max_id,
        model_dir=args.model_dir,
        save_checkpoint_epochs=10,
        log_steps=100,
        optimizer='adam',
        learning_rate=0.01,
        train_verbose=2,
        custom_metric_fn=custom_metric_fn,
    )
    trainer.train(**model_config)
    trainer.evaluate(**model_config)


if __name__ == "__main__":
    main()
