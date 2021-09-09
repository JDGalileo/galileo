---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform
---

# NAME

galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform
- transform for multi hop neighbors

# SYNOPSIS

\

继承自 **galileo.framework.python.base_transform.BaseTransform** .

被
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
, 以及
**galileo.framework.pytorch.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform**
继承.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, bool
edge_weight=False, \*\*kwargs)\

def **sample_multi_hop** (self, inputs)\
sample multi hop neighbors

def **transform** (self, inputs)\

## Public 属性

**fanouts_list**\

# 详细描述

transform for multi hop neighbors

**Examples:**

>     #without edge weight
>     >>> from galileo.pytorch import MultiHopNeighborTransform
>     >>> transform = MultiHopNeighborTransform([[0],[0]],[2,3]).transform
>     >>> res = transform([2,4])
>     >>> res['ids']
>     tensor([[   2, 1986, 1454, 1997,  899, 1987, 1454,    2, 1454],
>             [   4, 2176, 2176,    4,  595,    4, 1256, 2175,  595]])
>     >>> res['ids'].shape
>     torch.Size([2, 9])
>
>     #with edge weight
>     >>> transform = MultiHopNeighborTransform([[0],[0]],[2,3],True).transform
>     >>> res = transform([2,4])
>     >>> res['ids'].shape
>     torch.Size([2, 9])
>     >>> res['edge_weight'].shape
>     torch.Size([2, 9])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform.\_\_init\_\_ ( self, list metapath, list fanouts, bool edge_weight = `False`, \*\* kwargs)

**参数**

> *metapath* list of list, edge types of multi hop\
> *fanouts* number of multi hop\
> *edge_weight* has weight or not

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform.sample_multi_hop ( self, inputs)

sample multi hop neighbors

**参数**

> *inputs* vertices

## def galileo.framework.pytorch.python.transforms.multi_hop.MultiHopNeighborTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(ids=tensor, edge_weight=tensor)\
> may have duplicated vertices in ids

被
**galileo.framework.pytorch.python.transforms.multi_hop_feature_sparse.MultiHopFeatureSparseTransform**,
**galileo.framework.pytorch.python.transforms.multi_hop_feature_neg_sparse.MultiHopFeatureNegSparseTransform**,
**galileo.framework.pytorch.python.transforms.multi_hop_feature_neg.MultiHopFeatureNegTransform**,
**galileo.framework.pytorch.python.transforms.multi_hop_feature_label_sparse.MultiHopFeatureLabelSparseTransform**,
**galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform**
, 以及
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
重载.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
