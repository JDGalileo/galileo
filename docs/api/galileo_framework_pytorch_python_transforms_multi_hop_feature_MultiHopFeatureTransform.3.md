---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform
---

# NAME

galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform
- transform for multi hop features

# SYNOPSIS

\

继承自
**galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform**
.

被
**galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform**
, 以及
**galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform**
继承.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, bool
edge_weight=False, list dense_feature_names=None,
dense_feature_dims=None, list sparse_feature_names=None,
sparse_feature_dims=None, \*\*kwargs)\

def **transform** (self, inputs)\

def **get_feature** (self, vertices, feature_names, feature_dims,
feature_type, indices=None)\

## 额外继承的成员函数

# 详细描述

transform for multi hop features

**Examples:**

>     >>> from galileo.pytorch import MultiHopFeatureTransform
>     >>> transform = MultiHopFeatureTransform([[0],[0]],[2,3],
>             False,['feature'],5).transform
>     >>> res = transform([2,4])
>     >>> res.keys()
>     dict_keys(['ids', 'dense'])
>     >>> res['ids'].shape
>     torch.Size([2, 9])
>     >>> res['dense'].shape
>     torch.Size([2, 9, 5])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform.\_\_init\_\_ ( self, list metapath, list fanouts, bool edge_weight = `False`, list dense_feature_names = `None`, dense_feature_dims = `None`, list sparse_feature_names = `None`, sparse_feature_dims = `None`, \*\* kwargs)

**参数**

> *metapath* list of list, edge types of multi hop\
> *fanouts* number of multi hop\
> *edge_weight* has weight or not\
> *dense_feature_names* list of str\
> *dense_feature_dims* int or list\[int\]\
> *sparse_feature_names* list of str\
> *sparse_feature_dims* int or list\[int\]

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(ids=tensor, dense=tensor, sparse=tensor, edge_weight=tensor)

重载
**galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform**
.

被
**galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform**
, 以及
**galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform**
重载.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
