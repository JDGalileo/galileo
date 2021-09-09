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
Galileo KerasTrainer训练supervised graphsage模型
'''

import os
import argparse
import tensorflow as tf
import galileo as g
import galileo.tf as gt


class SupSAGE(gt.Supervised):
    def __init__(self,
                 hidden_dim,
                 num_classes,
                 dense_feature_dims,
                 fanouts,
                 aggregator_name='mean',
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_combiner = gt.FeatureCombiner(
            dense_feature_dims=dense_feature_dims)
        self.layer0 = gt.SAGELayer(hidden_dim,
                                   aggregator_name,
                                   activation='relu',
                                   dropout_rate=dropout_rate)
        self.layer1 = gt.SAGELayer(num_classes,
                                   aggregator_name,
                                   dropout_rate=dropout_rate)
        self.to_bipartite = gt.BipartiteTransform(fanouts).transform

    def encoder(self, inputs):
        feature = self.feature_combiner(inputs)
        bipartites = self.to_bipartite(dict(feature=feature))
        bipartites = self.layer0(bipartites)
        bipartites = self.layer1(bipartites)
        output = bipartites[-1]['src_feature']
        output = tf.squeeze(output)
        return output


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)
        self.transform = gt.MultiHopFeatureLabelTransform(
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
            outputs['target'] = inputs
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
    parser.add_argument('--label_dim',
                        default=7,
                        type=int,
                        help='label feature dimemsion')
    parser.add_argument('--model_dir',
                        default='.models/sup_sage_tf',
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
        num_classes=args.label_dim,
        dense_feature_dims=[args.dense_feature_dim],
        fanouts=fanouts,
        name='SupSAGE',
    )

    inputs = Inputs(vertex_type=[0],
                    metapath=[[0], [0]],
                    fanouts=fanouts,
                    label_name='label',
                    label_dim=args.label_dim,
                    dense_feature_names=['feature'],
                    dense_feature_dims=[args.dense_feature_dim],
                    data_source_name=args.data_source_name)

    trainer = gt.KerasTrainer(
        SupSAGE,
        inputs,
        model_args=model_args,
        distribution_strategy=args.ds,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
    )

    model_config = dict(
        batch_size=64,
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
