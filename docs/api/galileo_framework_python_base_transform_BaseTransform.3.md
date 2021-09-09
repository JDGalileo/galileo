---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_transform.BaseTransform
---

# NAME

galileo.framework.python.base_transform.BaseTransform - Base transform

# SYNOPSIS

\

继承自 **galileo.framework.python.named_object.NamedObject**, metaclass
, 以及 ABCMeta .

被
**galileo.framework.pytorch.python.transforms.bipartite.BipartiteTransform**,
**galileo.framework.pytorch.python.transforms.edge_neg.EdgeNegTransform**,
**galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform**,
**galileo.framework.pytorch.python.transforms.neighbor_neg.NeighborNegTransform**,
**galileo.framework.pytorch.python.transforms.relation.RelationTransform**,
**galileo.framework.pytorch.python.transforms.rw_neg.RandomWalkNegTransform**,
**galileo.framework.tf.python.transforms.bipartite.BipartiteTransform**,
**galileo.framework.tf.python.transforms.edge_neg.EdgeNegTransform**,
**galileo.framework.tf.python.transforms.multi_hop.MultiHopNeighborTransform**,
**galileo.framework.tf.python.transforms.neighbor_neg.NeighborNegTransform**,
**galileo.framework.tf.python.transforms.relation.RelationTransform** ,
以及
**galileo.framework.tf.python.transforms.rw_neg.RandomWalkNegTransform**
继承.

## Public 成员函数

def **\_\_init\_\_** (self, dict **config**=None, str name=None)\

def **config** (self)\
get config

def **transform** (self)\
subclass should override this method

# 详细描述

Base transform

# 构造及析构函数说明

## def galileo.framework.python.base_transform.BaseTransform.\_\_init\_\_ ( self, dict config = `None`, str name = `None`)

**参数**

> *config* dict, config\
> *name* name of inputs

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
