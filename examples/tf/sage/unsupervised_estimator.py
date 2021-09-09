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
'''
Galileo EstimatorTrainer训练unsupervised graphsage模型
'''

import os
import argparse
import tensorflow as tf
import galileo as g
import galileo.tf as gt


class SAGEEncode(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 dense_feature_dims,
                 fanouts,
                 aggregator_name='mean',
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dense_feature_dims = dense_feature_dims
        self.fanouts = fanouts
        self.aggregator_name = aggregator_name
        self.dropout_rate = dropout_rate
        self.feature_combiner = gt.FeatureCombiner(
            dense_feature_dims=dense_feature_dims)
        self.num_layers = len(fanouts)
        self.layers = [
            gt.SAGELayer(hidden_dim,
                         aggregator_name,
                         activation='relu',
                         dropout_rate=dropout_rate)
            for _ in range(self.num_layers - 1)
        ]
        self.layers.append(
            gt.SAGELayer(hidden_dim,
                         aggregator_name,
                         dropout_rate=dropout_rate))
        self.to_bipartite = gt.BipartiteTransform(fanouts).transform

    def call(self, inputs):
        feature = self.feature_combiner(inputs)
        bipartites = self.to_bipartite(dict(feature=feature))
        for layer in self.layers:
            bipartites = layer(bipartites)
        output = bipartites[-1]['src_feature']
        output = tf.squeeze(output, axis=2)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                hidden_dim=self.hidden_dim,
                dense_feature_dims=self.dense_feature_dims,
                fanouts=self.fanouts,
                aggregator_name=self.aggregator_name,
                dropout_rate=self.dropout_rate,
            ))
        return config


class UnsupSAGE(gt.Unsupervised):
    def __init__(
        self,
        hidden_dim,
        dense_feature_dims,
        fanouts,
        aggregator_name='mean',
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = SAGEEncode(
            hidden_dim,
            dense_feature_dims,
            fanouts,
            aggregator_name,
            dropout_rate,
        )

    def target_encoder(self, inputs):
        return self.encoder(inputs)

    def context_encoder(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                hidden_dim=self.hidden_dim,
                dense_feature_dims=self.dense_feature_dims,
                fanouts=self.fanouts,
                aggregator_name=self.aggregator_name,
                dropout_rate=self.dropout_rate,
            ))
        return config


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)
        self.transform = gt.MultiHopFeatureNegTransform(
            **self.config).transform

    def train_data(self):
        return gt.dataset_pipeline(gt.VertexDataset, self.transform,
                                   **self.config)

    def evaluate_data(self):
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gt.dataset_pipeline(
            lambda **kwargs: gt.TensorDataset(test_ids, **kwargs),
            self.transform, **self.config)

    def predict_data(self):
        def predict_transform(inputs):
            outputs = self.transform(inputs)
            outputs['target_ids'] = inputs
            return outputs

        return gt.dataset_pipeline(
            lambda **kwargs: gt.RangeDataset(
                start=0, end=kwargs['max_id'], **kwargs), predict_transform,
            **self.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_id', default=2708, type=int, help='max node id')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--ds',
                        default=None,
                        type=str,
                        help='distribution strategy '
                        '(mirrored, multi_worker_mirrored, parameter_server)')
    parser.add_argument('--dense_feature_dim',
                        default=1433,
                        type=int,
                        help='dense feature dimemsion')
    parser.add_argument('--model_dir',
                        default='.models/unsup_sage_tf',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fanouts = [5, 5]
    model_args = dict(
        hidden_dim=64,
        dense_feature_dims=[args.dense_feature_dim],
        fanouts=fanouts,
        metric_names='mrr',
        name='UnsupSAGE',
    )

    inputs = Inputs(vertex_type=[0],
                    metapath=[[0], [0]],
                    fanouts=fanouts,
                    negative_num=5,
                    dense_feature_names=['feature'],
                    dense_feature_dims=[args.dense_feature_dim],
                    data_source_name=args.data_source_name)

    trainer = gt.EstimatorTrainer(
        UnsupSAGE,
        inputs,
        model_args=model_args,
        distribution_strategy=args.ds,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
    )

    model_config = dict(
        batch_size=32,
        num_epochs=10,
        max_id=args.max_id,
        model_dir=args.model_dir,
        save_checkpoint_epochs=5,
        log_steps=100,
        optimizer='adam',
        learning_rate=0.01,
        train_verbose=2,
    )
    trainer.train(**model_config)
    trainer.evaluate(**model_config)
    trainer.predict(**model_config)


if __name__ == "__main__":
    main()
