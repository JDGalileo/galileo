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
使用Galileo EstimatorTrainer的ps模式训练Node2vec模型，使用多ps，PartitionedEmbedding
'''

import os
import argparse
import galileo as g
import galileo.tf as gt


class Node2vec(gt.Unsupervised):
    def __init__(self, embedding_size, embedding_dim, num_of_ps, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.num_of_ps = num_of_ps

        self._target_encoder = gt.PartitionedEmbedding(embedding_size,
                                                       embedding_dim,
                                                       num_of_ps)
        self._context_encoder = self._target_encoder

    def target_encoder(self, inputs):
        return self._target_encoder(inputs)

    def context_encoder(self, inputs):
        return self._context_encoder(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(embedding_size=self.embedding_size,
                 embedding_dim=self.embedding_dim,
                 num_of_ps=self.num_of_ps))
        return config


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)
        self.transform = gt.RandomWalkNegTransform(**self.config).transform

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
        return gt.dataset_pipeline(
            lambda **kwargs: gt.RangeDataset(
                start=0, end=kwargs['max_id'] + 1, **kwargs),
            lambda inputs: {'target': inputs}, **self.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_id', default=2708, type=int, help='max node id')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--model_dir',
                        default='.models/node2vec_tf',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inputs = Inputs(vertex_type=[0],
                    edge_types=[0],
                    walk_length=3,
                    repetition=5,
                    walk_p=1.,
                    walk_q=1.,
                    context_size=2,
                    negative_num=5,
                    data_source_name=args.data_source_name)

    model_args = dict(
        embedding_size=args.max_id + 1,
        embedding_dim=64,
        num_of_ps=2,
        metric_names='mrr',
        name='Node2vec',
    )
    trainer = gt.EstimatorTrainer(
        Node2vec,
        inputs,
        distribution_strategy='parameter_server',
        zk_server=args.zk_server,
        zk_path=args.zk_path,
        model_args=model_args,
    )

    model_config = dict(
        batch_size=32,
        max_id=args.max_id,
        model_dir=args.model_dir,
        num_epochs=10,
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
