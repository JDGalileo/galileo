---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform
---

# NAME

galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform
- transform for multi hop features with negative sampling

# SYNOPSIS

\

继承自
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, list
vertex_type, int negative_num, bool edge_weight=False, list
dense_feature_names=None, dense_feature_dims=None, list
sparse_feature_names=None, sparse_feature_dims=None, \*\*kwargs)\

def **transform** (self, inputs)\

def **reshape_outputs** (self, output, size)\

## 额外继承的成员函数

# 详细描述

transform for multi hop features with negative sampling

This is inputs for Unsupervised graphSAGE

**Examples:**

>     >>> from galileo.pytorch import MultiHopFeatureNegTransform
>     >>> transform = MultiHopFeatureNegTransform([[0],[0]],[2,3],[0],5,
>             False,['feature'],5).transform
>     >>> res = transform([2,4,6])
>     >>> res.keys()
>     dict_keys(['target', 'context', 'negative'])
>     >>> res['target'].keys()
>     dict_keys(['ids', 'dense'])
>     >>> res['target']['dense'].shape
>     torch.Size([3, 1, 9, 5])
>     >>> res['context']['dense'].shape
>     torch.Size([3, 1, 9, 5])
>     >>> res['negative']['dense'].shape
>     torch.Size([3, 5, 9, 5])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform.\_\_init\_\_ ( self, list metapath, list fanouts, list vertex_type, int negative_num, bool edge_weight = `False`, list dense_feature_names = `None`, dense_feature_dims = `None`, list sparse_feature_names = `None`, sparse_feature_dims = `None`, \*\* kwargs)

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

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(target=dict,context=dict,negative=dict)\
> inner dict: dict(ids=tensor, dense=tensor, sparse=tensor,
> edge_weight=tensor)

重载
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
