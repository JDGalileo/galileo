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
使用Galileo KerasTrainer训练LINE模型，高级用法
'''

import os
import argparse
import galileo as g
import galileo.tf as gt
import tensorflow as tf


class LINE(gt.Unsupervised):
    def __init__(self, embedding_size, embedding_dim, order=2, **kwargs):
        super().__init__(**kwargs)
        self._target_encoder = gt.Embedding(embedding_size, embedding_dim)
        if order == 1:
            self._context_encoder = self._target_encoder
        else:
            self._context_encoder = gt.Embedding(embedding_size, embedding_dim)

    def target_encoder(self, inputs):
        return self._target_encoder(inputs)

    def context_encoder(self, inputs):
        return self._context_encoder(inputs)


class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

    def train_transform(self, src, dst, types):
        vertex_type = self.config['vertex_type']
        negative_num = self.config['negative_num']
        target = tf.reshape(src, [-1, 1])
        size = target.shape[0]
        context = tf.reshape(dst, [-1, 1])
        negs = gt.ops.sample_vertices(vertex_type,
                                      count=size * negative_num)[0]
        negative = tf.reshape(negs, [size, negative_num])
        return {'target': target, 'context': context, 'negative': negative}

    def train_data(self):
        return gt.dataset_pipeline(gt.EdgeDataset, self.train_transform,
                                   **self.config)

    def eval_transform(self, vertices):
        vertex_type = self.config['vertex_type']
        edge_types = self.config['edge_types']
        negative_num = self.config['negative_num']
        size = tf.size(vertices)
        target = tf.reshape(vertices, [-1, 1])
        positive_ = gt.ops.sample_neighbors(tf.reshape(vertices, [-1]),
                                            edge_types,
                                            count=1,
                                            has_weight=False)
        context = tf.reshape(positive_, [-1, 1])
        negs = gt.ops.sample_vertices(vertex_type,
                                      count=size * negative_num)[0]
        negative = tf.reshape(negs, [size, negative_num])
        return {'target': target, 'context': context, 'negative': negative}

    def evaluate_data(self):
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gt.dataset_pipeline(
            lambda **kwargs: gt.TensorDataset(test_ids, **kwargs),
            self.eval_transform, **self.config)

    def predict_data(self):
        return gt.dataset_pipeline(
            lambda **kwargs: gt.RangeDataset(
                start=0, end=kwargs['max_id'] + 1, **kwargs),
            lambda inputs: {'target': inputs}, **self.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_id', default=2708, type=int, help='max node id')
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--order', default=2, type=int, help='LINE order')
    parser.add_argument('--model_dir',
                        default='.models/line_tf',
                        type=str,
                        help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_args = dict(
        embedding_size=args.max_id + 1,
        embedding_dim=64,
        order=args.order,
        metric_names='Mrr',
        name='LINE',
    )

    inputs = Inputs(vertex_type=[0],
                    edge_types=[0],
                    negative_num=5,
                    data_source_name=args.data_source_name)

    is_multi_gpu = len(args.gpu.split(',')) > 1
    ds = 'mirrored' if is_multi_gpu else None
    trainer = gt.KerasTrainer(
        LINE,
        inputs,
        distribution_strategy=ds,
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
