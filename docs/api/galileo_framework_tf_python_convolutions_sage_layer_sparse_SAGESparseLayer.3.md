---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer
---

# NAME

galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer
- graphSAGE convolution tf layer, sparse version

# SYNOPSIS

\

继承自 Layer , 以及
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## Public 成员函数

def **\_\_init\_\_** (self, int output_dim, str
aggregator_name=\'mean\', bool use_concat_in_aggregator=True, bool
bias=True, float dropout_rate=0.0, activation=None, normalization=None,
\*\*kwargs)\

def **call** (self, inputs, training=None)\

def **message_and_aggregate** (self, inputs, training=None)\
\"message*and*aggregate*features*on*vertices*and*edges*\
\"

def **update** (self, inputs)\
\"update*target*features\
\"

def **get_config** (self)\

## Public 属性

**output_dim**\

**aggregator_name**\

**use_concat_in_aggregator**\

**bias**\

**dropout_rate**\

**activation**\

**normalization**\

**aggregator**\

**feature_dropout**\

# 详细描述

graphSAGE convolution tf layer, sparse version

`'Inductive Representation Learning on Large Graphs' <https://arxiv.org/abs/1706.02216>`

# 构造及析构函数说明

## def galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer.\_\_init\_\_ ( self, int output_dim, str aggregator_name = `'mean'`, bool use_concat_in_aggregator = `True`, bool bias = `True`, float dropout_rate = `0.0`, activation = `None`, normalization = `None`, \*\* kwargs)

**参数**

> *output_dim* output dim of layer\
> *aggregator_name* aggregator name, one of \'mean, mean-1k, mean-2k,
> gcn, meanpool, maxpool\'\
> *use_concat_in_aggregator* concat if True else sum when aggregate\
> *bias* bias of layer\
> *dropout_rate* feature dropout rate\
> *activation* callable, apply activation to the updated vertices
> features\
> *normalization* callable, apply normalization to the updated vertices
> features

# 成员函数说明

## def galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer.call ( self, inputs, training = `None`)

**参数**

> *inputs* relation graph dict( relation_indices=tensor, feature=tensor,
> relation_weight=tensor, )

## def galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer.message_and_aggregate ( self, inputs, training = `None`)

message and aggregate features on vertices and edges\
subclass should override this method

**参数**

> *inputs* inputs for message passing\
> *training*

**返回**

> tensors

重载
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## def galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer.update ( self, inputs)

update target features\
subclass should override this method

**参数**

> *inputs* inputs for update

**返回**

> tensors

重载
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
