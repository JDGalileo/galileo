import os
import argparse
import numpy as np
import galileo as g
import galileo.tf as gt
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import utils
import functools


class Inputs(g.BaseInputs):

    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

    def transform(self, vertices, mode='train'):
        vertices = tf.cast(vertices, tf.int64)
        size = tf.size(vertices)
        vertices = tf.reshape(vertices, (size, ))
        last_neighbors_size = np.array([1] +
                                       self.config['fanouts']).cumprod()[-1]
        neighbors_feats = []
        for metapath in self.config['metapath_groups']:
            multi_hops = gt.ops.sample_seq_by_multi_hop(
                vertices=vertices,
                metapath=metapath,
                fanouts=self.config['fanouts'],
                has_weight=False)[0]
            multi_hops = multi_hops[:, -last_neighbors_size:]
            # multi_hops.shape:[batch_size, last_neighbors_size]
            _mp_neighbor_features = gt.ops.get_pod_feature(
                [tf.reshape(multi_hops, [-1])], [self.config['feature_name']],
                self.config['feature_dim'], [tf.float32])[0]
            mp_neighbor_features = tf.reshape(
                _mp_neighbor_features,
                [1, -1, last_neighbors_size, self.config['feature_dim'][0]])
            # mp_neighbor_features.shape:[1, batch_size, last_neighbors_size, dim]
            neighbors_feats.append(mp_neighbor_features)
        neighbors_feats = tf.concat(
            neighbors_feats, axis=0
        )  # neighbors_feats.shape:[path_num, batch_size, last_neighbors_size, dim]
        _node_feats = gt.ops.get_pod_feature([vertices],
                                             [self.config['feature_name']],
                                             self.config['feature_dim'],
                                             [tf.float32])[0]
        node_feats = tf.reshape(_node_feats,
                                [-1, self.config['feature_dim'][0]
                                 ])  # node_feats.shape:[batch_size, dim]

        if mode == 'train':
            _labels = gt.ops.get_pod_feature([vertices],
                                             [self.config['label_name']],
                                             self.config['label_dim'],
                                             [tf.int64])[0]
            _labels = tf.one_hot(_labels, self.config['num_labels'])
            labels = tf.cast(tf.reshape(_labels,
                                        [-1, self.config['num_labels']]),
                             dtype=tf.float32)
            # labels.shape:[batch_size, num_labels]
            return dict(ids=vertices,
                        targets=node_feats,
                        contexts=neighbors_feats,
                        labels=labels)
        else:
            return dict(ids=vertices,
                        targets=node_feats,
                        contexts=neighbors_feats)

    def train_data(self):

        def base_dataset(**kwargs):
            train_idx = self.config['data_source'].get_train_data()
            ds = tf.data.Dataset.from_tensor_slices(train_idx).shuffle(
                self.config['batch_size'] * 5).repeat()
            return ds

        return gt.dataset_pipeline(base_dataset, self.transform, **self.config)

    def evaluate_data(self):

        def base_dataset(**kwargs):
            eval_idx = self.config['data_source'].get_eval_data()
            ds = tf.data.Dataset.from_tensor_slices(eval_idx).shuffle(
                self.config['batch_size'] * 5)
            return ds

        return gt.dataset_pipeline(base_dataset, self.transform, **self.config)

    def predict_data(self):

        def base_dataset(**kwargs):
            test_idx = self.config['data_source'].get_test_data()
            ds = tf.data.Dataset.from_tensor_slices(test_idx).shuffle(
                self.config['batch_size'] * 5)
            return ds

        return gt.dataset_pipeline(
            base_dataset, functools.partial(self.transform, mode='predict'),
            **self.config)


class SemanticAttention(tf.keras.layers.Layer):

    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def get_config(self):
        config = super().get_config()
        config.update(dict(attention_dim=self.attention_dim))
        return config

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],
                                             self.attention_dim),
                                      initializer='glorot_normal')
        self.att_kernel = self.add_weight(name='att_kernel',
                                          shape=(self.attention_dim, 1),
                                          initializer='glorot_normal')
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape [batch size, groups length, dim]
        inputs = tf.einsum('GBD->BGD', inputs)
        groups_len = inputs.shape[-2]
        att_scores = tf.nn.softmax(
            tf.reshape(
                tf.matmul(tf.tanh(tf.matmul(inputs, self.kernel)),
                          self.att_kernel), [-1, groups_len]))
        att_scores = tf.reshape(att_scores, [-1, 1, groups_len])
        output = tf.matmul(att_scores, inputs)
        output = tf.squeeze(output, axis=1)  # output shape [batch size, dim]
        return output


class NodeAttention(tf.keras.layers.Layer):

    def __init__(self,
                 channels,
                 heads=1,
                 concat=True,
                 drop_out=0.6,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.heads = heads
        self.concat = concat
        self.dropout_rate = drop_out
        self.activation = activation
        self.use_bias = use_bias

        if self.concat:
            self.output_dim = self.heads * self.channels
        else:
            self.output_dim = self.channels

    def get_config(self):
        config = {
            "heads": self.heads,
            "concat": self.concat,
            "drop_out": self.dropout
        }
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

    def build(self, batch_input_shape):
        input_dim = batch_input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.channels],
            initializer='glorot_uniform',
            regularizer=l2(5e-6),
        )
        self.attn_kernel = self.add_weight(
            name="attn_kernel",
            shape=[1, self.heads, 2 * self.channels],
            initializer='glorot_uniform',
            regularizer=l2(5e-6))
        self.attn_bias = self.add_weight(name="attn_bias",
                                         shape=[self.output_dim],
                                         initializer='zeros')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.built = True

    def call(self, node_feats, neighbors_feats):
        # node_feats shape [batch_size, dim]
        # neighbors_feats shape [batch_size, neighbor, dim]
        _node_feats, _neighbors_feats = node_feats, neighbors_feats
        node_feats = tf.einsum('BD,DC->BC', _node_feats, self.kernel)
        neighbors_feats = tf.einsum('BND, DC->BNC', _neighbors_feats,
                                    self.kernel)
        neighbor_size = neighbors_feats.shape[1]
        tile_node_feats = tf.tile(
            tf.reshape(node_feats, [-1, 1, self.channels]),
            [1, neighbor_size, 1])
        concat_node_feats = tf.concat([tile_node_feats, neighbors_feats],
                                      axis=2)
        res = tf.einsum('...NO, ...HO->...NH', concat_node_feats,
                        self.attn_kernel)  # BATCH, NEIGHBOR, HEADS
        coef = self.dropout(
            tf.nn.softmax(tf.nn.leaky_relu(tf.einsum('BNH->BHN', res))))
        _z_embeddings = tf.einsum('BHN, BNO-> BHO', coef, neighbors_feats)
        if self.concat:
            shape = _z_embeddings.shape[:-2] + [self.heads * self.channels]
            shape = [d if d is not None else -1 for d in shape]
            _z_embeddings = tf.reshape(_z_embeddings, shape)
        else:
            _z_embeddings = tf.reduce_mean(_z_embeddings, axis=-2)
        if self.use_bias:
            _z_embeddings += self.attn_bias
        # _z_embeddings = self.activation(_z_embeddings)
        z_embeddings = tf.expand_dims(_z_embeddings, axis=0)
        return z_embeddings


class HAN(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop('name'))
        self.metapath_groups = kwargs.pop('metapath_groups')
        self.sem_attention_dim = kwargs.pop('sem_attention_dim')
        self.node_attention_dim = kwargs.pop('node_attention_dim')
        self.attention_heads = kwargs.pop('attention_heads')
        self.num_labels = kwargs.pop('num_labels')
        self.NodeAttention = NodeAttention(channels=self.node_attention_dim,
                                           heads=self.attention_heads)
        self.SemanticAttention = SemanticAttention(
            attention_dim=self.sem_attention_dim)
        self.dense = tf.keras.layers.Dense(self.num_labels)

    def get_config(self):
        config = super().get_config()
        config.update(metapath_groups=self.metapath_groups)
        return config

    def call(self, inputs):
        node_feats = inputs.get('targets')
        neighbors_feats = inputs.get('contexts')
        z_embeddings = []
        for i in range(neighbors_feats.shape[0]):
            z_embeddings.append(
                self.NodeAttention(node_feats, neighbors_feats[i]))
        z_embeddings = tf.concat(
            z_embeddings, axis=0
        )  # z_embeddings.shape:[path_num, batch_size, node_attention_dim*attention_heads]
        z = self.SemanticAttention(
            z_embeddings
        )  # z.shape:[batch_size, node_attention_dim*attention_heads]
        logits = self.dense(z)
        outputs = dict()
        outputs['logits'] = logits  # logits.shape:[batch_size,1]
        if 'labels' in inputs:
            outputs['labels'] = inputs['labels']
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=inputs['labels'], logits=logits)
            loss = tf.reduce_mean(loss)
            outputs['loss'] = loss
            self.add_loss(loss)
        else:
            outputs = dict()
            outputs['logits'] = logits  # logits.shape:[batch_size,1]
            outputs['ids'] = inputs.get('ids')

        print('han outputs:', outputs)
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='gpu devices')
    parser.add_argument('--trainer', default='keras', type=str, help='trainer')
    parser.add_argument('--ds',
                        default=None,
                        type=str,
                        help='distribution strategy '
                        '(mirrored, multi_worker_mirrored, parameter_server)')
    parser.add_argument('--model_dir',
                        default='.models/HAN_tf',
                        type=str,
                        help='model dir')
    parser.add_argument('--data-source',
                        '-i',
                        default='acm',
                        type=str,
                        help='only acm')
    parser.add_argument('--debug', '-d', action='store_true')

    parser = g.define_service_args(parser)
    args, _ = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_source = utils.DataSource(args.data_source)
    args.data_path = data_source.output_dir
    g.start_service_from_args(args)

    fanouts = [3, 3]
    node_attention_dim = 8
    sem_attention_dim = 128
    attention_heads = 8
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.005
    feature_dim = 1903
    max_id = 4025
    label_dim = 1
    num_labels = 3
    metapath_groups_0 = [[0], [1]]
    metapath_groups_1 = [[2], [3]]
    metapath_groups = [metapath_groups_0, metapath_groups_1]
    inputs = Inputs(
        data_source=data_source,
        vertex_type=[0],
        batch_size=batch_size,
        num_epochs=num_epochs,
        metapath_groups=metapath_groups,
        fanouts=fanouts,
        label_name='label',
        label_dim=[label_dim],
        num_labels=num_labels,
        feature_name='feature',
        feature_dim=[feature_dim],
    )

    model_args = dict(
        metapath_groups=metapath_groups,
        fanouts=fanouts,
        max_id=max_id,
        sem_attention_dim=sem_attention_dim,
        node_attention_dim=node_attention_dim,
        attention_heads=attention_heads,
        feature_name='feature',
        feature_dim=[feature_dim],
        num_labels=num_labels,
        name='HAN',
    )

    is_multi_gpu = len(args.gpu.split(',')) > 1
    trainer = gt.EstimatorTrainer(
        HAN,
        inputs,
        distribution_strategy='mirrored' if is_multi_gpu else None,
        zk_server=args.zk_server,
        zk_path=args.zk_path,
        model_args=model_args,
    )

    model_config = dict(
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_id=max_id,
        model_dir=args.model_dir,
        save_checkpoint_epochs=1,
        log_steps=100,
        optimizer='adam',
        learning_rate=learning_rate,
        train_verbose=1,
    )
    trainer.train(**model_config)
    # model_config['batch_size'] = 128
    outputs = trainer.predict(**model_config)[0]
    tm = utils.compute_test_metrics(data_source, outputs)
    print(tm)


if __name__ == "__main__":
    main()
