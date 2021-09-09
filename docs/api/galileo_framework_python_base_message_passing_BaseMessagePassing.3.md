---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_message_passing.BaseMessagePassing
---

# NAME

galileo.framework.python.base_message_passing.BaseMessagePassing - Base
message passing for GNN

# SYNOPSIS

\

继承自 **galileo.framework.python.named_object.NamedObject**, metaclass
, 以及 ABCMeta .

被
**galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer**,
**galileo.framework.pytorch.python.convolutions.sage_layer_sparse.SAGESparseLayer**,
**galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer**,
**galileo.framework.tf.python.convolutions.sage_layer.SAGELayer** , 以及
**galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer**
继承.

## Public 成员函数

def **\_\_init\_\_** (self, dict **config**=None, str name=None)\

def **config** (self)\
get config

def **\_\_call\_\_** (self, inputs, training=None)\

def **message** (self, inputs, training=None)\
\"message*features*on*vertices*and*edges*\
\"

def **aggregate** (self, inputs)\
\"aggregate*messages*from*neighbors*with*target*\
\"

def **update** (self, inputs)\
\"update*target*features\
\"

def **message_and_aggregate** (self, inputs, training=None)\
\"message*and*aggregate*features*on*vertices*and*edges*\
\"

# 详细描述

Base message passing for GNN

paper
`'Neural Message Passing for Quantum Chemistry' <https://arxiv.org/abs/1704.01212>`

# 构造及析构函数说明

## def galileo.framework.python.base_message_passing.BaseMessagePassing.\_\_init\_\_ ( self, dict config = `None`, str name = `None`)

**参数**

> *config* dict, config\
> *name* name of inputs

# 成员函数说明

## def galileo.framework.python.base_message_passing.BaseMessagePassing.aggregate ( self, inputs)

aggregate messages from neighbors with target\
subclass should override this method

**参数**

> *inputs* inputs for aggregate

**返回**

> tensors

被 **galileo.framework.tf.python.convolutions.sage_layer.SAGELayer** ,
以及
**galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer**
重载.

## def galileo.framework.python.base_message_passing.BaseMessagePassing.message ( self, inputs, training = `None`)

message features on vertices and edges\
subclass should override this method

**参数**

> *inputs* inputs for message passing\
> *training*

**返回**

> tensors

被 **galileo.framework.tf.python.convolutions.sage_layer.SAGELayer** ,
以及
**galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer**
重载.

## def galileo.framework.python.base_message_passing.BaseMessagePassing.message_and_aggregate ( self, inputs, training = `None`)

message and aggregate features on vertices and edges\
subclass should override this method

**参数**

> *inputs* inputs for message passing\
> *training*

**返回**

> tensors

被
**galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer**,
**galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer** , 以及
**galileo.framework.pytorch.python.convolutions.sage_layer_sparse.SAGESparseLayer**
重载.

## def galileo.framework.python.base_message_passing.BaseMessagePassing.update ( self, inputs)

update target features\
subclass should override this method

**参数**

> *inputs* inputs for update

**返回**

> tensors

被
**galileo.framework.tf.python.convolutions.sage_layer_sparse.SAGESparseLayer**,
**galileo.framework.tf.python.convolutions.sage_layer.SAGELayer**,
**galileo.framework.tf.python.convolutions.gcn_layer.GCNLayer**,
**galileo.framework.pytorch.python.convolutions.sage_layer_sparse.SAGESparseLayer**
, 以及
**galileo.framework.pytorch.python.convolutions.sage_layer.SAGELayer**
重载.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
