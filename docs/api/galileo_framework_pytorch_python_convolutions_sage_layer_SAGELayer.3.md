---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer
---

# NAME

galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer -
graphSAGE convolution pytorch layer

# SYNOPSIS

\

继承自 nn.Module , 以及
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## Public 成员函数

def **\_\_init\_\_** (self, int input_dim, int output_dim, str
aggregator_name=\'mean\', bool use_concat_in_aggregator=True, bool
bias=True, float dropout_rate=0.0, activation=None, normalization=None)\

def **forward** (self, inputs, training=None)\

def **message** (self, inputs, training=None)\
\"message*features*on*vertices*and*edges*\
\"

def **aggregate** (self, inputs)\
\"aggregate*messages*from*neighbors*with*target*\
\"

def **update** (self, inputs)\
\"update*target*features\
\"

## Public 属性

**input_dim**\

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

graphSAGE convolution pytorch layer

`'Inductive Representation Learning on Large Graphs' <https://arxiv.org/abs/1706.02216>`

# 构造及析构函数说明

## def galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer.\_\_init\_\_ ( self, int input_dim, int output_dim, str aggregator_name = `'mean'`, bool use_concat_in_aggregator = `True`, bool bias = `True`, float dropout_rate = `0.0`, activation = `None`, normalization = `None`)

**参数**

> *input_dim* input dim of layer\
> *output_dim* output dim of layer\
> *aggregator_name* aggregator name, one of \'mean, gcn, meanpool,
> maxpool\'\
> *use_concat_in_aggregator* concat if True else sum when aggregate\
> *bias* bias of layer\
> *dropout_rate* feature dropout rate\
> *activation* callable, apply activation to the updated vertices
> features\
> *normalization* callable, apply normalization to the updated vertices
> features

# 成员函数说明

## def galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer.aggregate ( self, inputs)

aggregate messages from neighbors with target\
subclass should override this method

**参数**

> *inputs* inputs for aggregate

**返回**

> tensors

重载
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## def galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer.forward ( self, inputs, training = `None`)

**参数**

> *inputs* tensors of inputs shape (batch_size, \*, fanouts,
> feature_dim)

## def galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer.message ( self, inputs, training = `None`)

message features on vertices and edges\
subclass should override this method

**参数**

> *inputs* inputs for message passing\
> *training*

**返回**

> tensors

重载
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## def galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer.update ( self, inputs)

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
