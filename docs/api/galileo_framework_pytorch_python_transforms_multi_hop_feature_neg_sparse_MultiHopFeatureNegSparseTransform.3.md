---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform
---

# NAME

galileo.framework.pytorch.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform
- transform for multi hop features with negative sampling, sparse
version

# SYNOPSIS

\

继承自
**galileo.framework.pytorch.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform**
.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, list
vertex_type, int negative_num, bool edge_weight=False, list
dense_feature_names=None, dense_feature_dims=None, list
sparse_feature_names=None, sparse_feature_dims=None, \*\*kwargs)\

def **transform** (self, inputs)\

## 额外继承的成员函数

# 详细描述

transform for multi hop features with negative sampling, sparse version

This is inputs for Unsupervised graphSAGE

**Examples:**

>     >>> from galileo.pytorch import MultiHopFeatureNegSparseTransform
>     >>> transform = MultiHopFeatureNegSparseTransform([[0],[0]],[2,3],[0],5,
>             False,['feature'],5).transform
>     >>> res = transform([2,4,6])
>     >>> res.keys()
>     dict_keys(['target', 'context', 'negative'])
>     >>> res['target'].keys()
>     dict_keys(['ids', 'indices', 'dense'])
>     >>> res['target']['ids'].shape
>     torch.Size([22])
>     >>> res['target']['indices'].shape
>     torch.Size([3, 1, 9])
>     >>> res['context']['dense'].shape
>     torch.Size([17, 5])
>     >>> res['negative']['ids'].shape
>     torch.Size([78])
>     >>> res['negative']['dense'].shape
>     torch.Size([78, 5])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform.\_\_init\_\_ ( self, list metapath, list fanouts, list vertex_type, int negative_num, bool edge_weight = `False`, list dense_feature_names = `None`, dense_feature_dims = `None`, list sparse_feature_names = `None`, sparse_feature_dims = `None`, \*\* kwargs)

**参数**

> *metapath* list of list, edge types of multi hop\
> *fanouts* number of multi hop\
> *vertex_type* vertex type\
> *negative_num* number of negative\
> *edge_weight* has weight or not\
> *dense_feature_names* list of str\
> *dense_feature_dims* int or list\[int\]\
> *sparse_feature_names* list of str\
> *sparse_feature_dims* int or list\[int\]

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(target=dict,context=dict,negative=dict)\
> inner dict: dict(ids=tensor, indices=tensor, dense=tensor,
> sparse=tensor, edge_weight=tensor)

重载
**galileo.framework.pytorch.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform**
.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
