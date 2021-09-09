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
Galileo训练unsupervised graphsage模型的简单用法
'''

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import galileo as g
import galileo.pytorch as gp


class SAGEEncode(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dense_feature_dims,
        fanouts,
        aggregator_name='mean',
        dropout_rate=0.0,
    ):
        super().__init__()
        self.feature_combiner = gp.FeatureCombiner(
            dense_feature_dims=dense_feature_dims)
        self.layer0 = gp.SAGESparseLayer(input_dim,
                                         output_dim,
                                         aggregator_name,
                                         activation=F.relu,
                                         dropout_rate=dropout_rate)
        self.layer1 = gp.SAGESparseLayer(output_dim,
                                         output_dim,
                                         aggregator_name,
                                         dropout_rate=dropout_rate)
        self.relation = gp.RelationTransform(fanouts).transform

    def forward(self, inputs):
        feature = self.feature_combiner(inputs)
        relation_graph = self.relation(inputs)
        relation_graph['feature'] = feature
        feature = self.layer0(relation_graph)
        relation_graph['feature'] = feature
        feature = self.layer1(relation_graph)
        ti = relation_graph['target_indices']
        output = feature.index_select(0, ti.flatten())
        return output.reshape(ti.shape + (feature.shape[-1], ))


class UnsupSAGE(gp.Unsupervised):
    def __init__(
        self,
        input_dim,
        output_dim,
        dense_feature_dims,
        fanouts,
        aggregator_name='mean',
        dropout_rate=0.0,
    ):
        super().__init__()
        self.encoder = SAGEEncode(
            input_dim,
            output_dim,
            dense_feature_dims,
            fanouts,
            aggregator_name,
            dropout_rate,
        )

    def target_encoder(self, inputs):
        return self.encoder(inputs)

    def context_encoder(self, inputs):
        return self.encoder(inputs)


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)
        self.transform = gp.MultiHopFeatureNegSparseTransform(
            **self.config).transform

    def train_data(self):
        return gp.dataset_pipeline(gp.VertexDataset, self.transform,
                                   **self.config)

    def evaluate_data(self):
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gp.dataset_pipeline(
            lambda **kwargs: gp.TensorDataset(test_ids, **kwargs),
            self.transform, **self.config)

    def predict_data(self):
        def predict_transform(inputs):
            inputs = torch.tensor(inputs)
            outputs = self.transform(inputs)
            outputs['target_ids'] = inputs
            return outputs

        return gp.dataset_pipeline(
            lambda **kwargs: gp.RangeDataset(
                start=0, end=kwargs['max_id'], **kwargs), predict_transform,
            **self.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_id', default=2708, type=int, help='max node id')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--model_dir',
                        default='.models/unsup_sage_pt',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sage = UnsupSAGE(
        input_dim=1433,
        output_dim=64,
        dense_feature_dims=[1433],
        fanouts=[5, 5],
    )

    inputs = Inputs(batch_size=64,
                    vertex_type=[0],
                    metapath=[[0], [0]],
                    fanouts=[5, 5],
                    negative_num=5,
                    dense_feature_names=['feature'],
                    dense_feature_dims=[1433],
                    max_id=args.max_id,
                    data_source_name=args.data_source_name)

    is_multi_gpu = len(args.gpu.split(',')) > 1
    trainer = gp.Trainer(
        sage,
        inputs,
        multiprocessing_distributed=is_multi_gpu,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
    )

    model_config = dict(
        model_dir=args.model_dir,
        num_epochs=10,
        save_checkpoint_epochs=5,
        log_steps=100,
        optimizer='adam',
        learning_rate=0.01,
    )
    trainer.train(**model_config)
    trainer.evaluate(**model_config)
    trainer.predict(**model_config)


if __name__ == "__main__":
    main()
