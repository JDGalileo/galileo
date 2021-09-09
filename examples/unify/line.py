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
使用Galileo统一的接口训练LINE模型
'''

import os
import argparse
import galileo as g
from galileo.unify import api as gu_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        default='pytorch',
                        type=str,
                        help='backend')
    parser.add_argument('--max_id', default=2708, type=int, help='max node id')
    parser.add_argument('--order', default=2, type=int, help='LINE order')
    parser.add_argument('--model_dir',
                       default='.models/line_unify',
                       type=str,
                       help='model dir')
    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()
    if args.data_source_name is None:
        args.data_source_name = 'cora'
    g.start_service_from_args(args)

    gu = gu_api(backend=args.backend)

    class Inputs(g.BaseInputs):
        def __init__(self, **kwargs):
            super().__init__(config=kwargs)
            self.train_transform = gu.EdgeNegTransform(**self.config).transform
            self.evaluate_transform = gu.NeighborNegTransform(
                **self.config).transform

        def train_data(self):
            return gu.dataset_pipeline(gu.EdgeDataset, self.train_transform,
                                       **self.config)

        def evaluate_data(self):
            test_ids = g.get_test_vertex_ids(
                data_source_name=self.config['data_source_name'])
            return gu.dataset_pipeline(
                lambda **kwargs: gu.TensorDataset(test_ids, **kwargs),
                self.evaluate_transform, **self.config)

        def predict_data(self):
            return gu.dataset_pipeline(
                lambda **kwargs: gu.RangeDataset(
                    start=0, end=kwargs['max_id'] + 1, **kwargs),
                lambda inputs: {'target': inputs}, **self.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    line = gu.VertexEmbedding(
        embedding_size=args.max_id + 1,
        embedding_dim=64,
        shared_embeddings=args.order == 1,
    )

    inputs = Inputs(batch_size=32,
                    max_id=args.max_id,
                    vertex_type=[0],
                    edge_types=[0],
                    negative_num=5,
                    data_source_name=args.data_source_name)

    trainer = gu.Trainer(
        line,
        inputs,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
    )

    model_config = dict(
        batch_size=32,
        max_id=args.max_id,
        model_dir=args.model_dir,
        num_epochs=10,
        save_checkpoint_epochs=5,
        log_steps=100,
        optimizer='adam',
        learning_rate=0.05,
    )

    trainer.train(**model_config)
    trainer.evaluate(**model_config)
    trainer.predict(**model_config)


if __name__ == "__main__":
    main()
