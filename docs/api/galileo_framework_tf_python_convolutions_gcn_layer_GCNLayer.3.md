---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer
---

# NAME

galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer - GCN
convolution tf layer, sparse version

# SYNOPSIS

\

继承自 Layer , 以及
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## Public 成员函数

def **\_\_init\_\_** (self, int output_dim, bool bias=True, float
dropout_rate=0.0, activation=None, normalization=None, \*\*kwargs)\

def **build** (self, input_shape)\

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

**bias**\

**dropout_rate**\

**activation**\

**normalization**\

**feature_dropout**\

**kernels**\

**biases**\

# 详细描述

GCN convolution tf layer, sparse version

`'Semi-Supervised Classification with Graph Convolutional Networks' <https://arxiv.org/abs/1609.02907>`

# 构造及析构函数说明

## def galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer.\_\_init\_\_ ( self, int output_dim, bool bias = `True`, float dropout_rate = `0.0`, activation = `None`, normalization = `None`, \*\* kwargs)

**参数**

> *output_dim* output dim of layer\
> *bias* bias of layer\
> *dropout_rate* feature dropout rate\
> *activation* callable, apply activation to the updated vertices
> features\
> *normalization* callable, apply normalization to the updated vertices
> features

# 成员函数说明

## def galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer.message_and_aggregate ( self, inputs, training = `None`)

message and aggregate features on vertices and edges\
subclass should override this method

**参数**

> *inputs* inputs for message passing\
> *training*

**返回**

> tensors

重载
**galileo.framework.python.base_message_passing.BaseMessagePassing** .

## def galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer.update ( self, inputs)

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
