---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform
---

# NAME

galileo.framework.tf.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform
- transform for multi hop features, sparse version

# SYNOPSIS

\

继承自
**galileo.framework.tf.python.transforms.multi_hop.MultiHopNeighborTransform**
.

被
**galileo.framework.tf.python.transforms.multi_hop_feature_label_sparse.MultiHopFeatureLabelSparseTransform**
, 以及
**galileo.framework.tf.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform**
继承.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, bool
edge_weight=False, list dense_feature_names=None,
dense_feature_dims=None, list sparse_feature_names=None,
sparse_feature_dims=None, \*\*kwargs)\

def **transform** (self, inputs)\

def **get_feature** (self, vertices, feature_names, feature_dims,
feature_type)\

## Public 属性

**fanouts_dim**\

# 详细描述

transform for multi hop features, sparse version

**Examples:**

>     >>> from galileo.tf import MultiHopFeatureSparseTransform
>     >>> transform = MultiHopFeatureSparseTransform([[0],[0]],[2,3],
>             False,['feature'],5).transform
>     >>> res = transform([2,4])
>     >>> res.keys()
>     dict_keys(['ids', 'indices', 'dense'])
>     >>> res['ids'].shape
>     TensorShape([10])
>     >>> res['indices'].shape
>     TensorShape([2, 9])
>     >>> res['dense'].shape
>     TensorShape([10, 5])
>
>     >>> transform = MultiHopFeatureSparseTransform([[0],[0]],[2,3],
>             True,['feature'],5).transform
>     >>> res = transform([[2],[4],[8]])
>     >>> res.keys()
>     dict_keys(['ids', 'indices', 'dense', 'edge_weight'])
>     >>> res['ids'].shape
>     TensorShape([17])
>     >>> res['indices'].shape
>     TensorShape([3, 1, 9])
>     >>> res['dense'].shape
>     TensorShape([17, 5])
>     >>> res['edge_weight'].shape
>     TensorShape([3, 9])

# 构造及析构函数说明

## def galileo.framework.tf.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform.\_\_init\_\_ ( self, list metapath, list fanouts, bool edge_weight = `False`, list dense_feature_names = `None`, dense_feature_dims = `None`, list sparse_feature_names = `None`, sparse_feature_dims = `None`, \*\* kwargs)

**参数**

> *metapath* list of list, edge types of multi hop\
> *fanouts* number of multi hop\
> *edge_weight* has weight or not\
> *dense_feature_names* list of str\
> *dense_feature_dims* int or list\[int\]\
> *sparse_feature_names* list of str\
> *sparse_feature_dims* int or list\[int\]

# 成员函数说明

## def galileo.framework.tf.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(ids=tensor, indices=tensor, dense=tensor, sparse=tensor,
> edge_weight=tensor)\
>
> -   ids shape \[U\]
>
> -   indices shape inputs.shape + fanouts_dim
>
> -   dense sparse shape \[U, dim\]
>
> -   edge_weight shape \[N, fanouts_dim\]

重载
**galileo.framework.tf.python.transforms.multi_hop.MultiHopNeighborTransform**
.

被
**galileo.framework.tf.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform**
, 以及
**galileo.framework.tf.python.transforms.multi_hop_feature_label_sparse.MultiHopFeatureLabelSparseTransform**
重载.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
