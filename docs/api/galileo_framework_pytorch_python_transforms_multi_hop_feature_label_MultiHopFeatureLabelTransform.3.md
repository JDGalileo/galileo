---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform
---

# NAME

galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform
- transform for multi hop features and label

# SYNOPSIS

\

继承自
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
.

## Public 成员函数

def **\_\_init\_\_** (self, list metapath, list fanouts, str label_name,
int label_dim, bool edge_weight=False, list dense_feature_names=None,
dense_feature_dims=None, list sparse_feature_names=None,
sparse_feature_dims=None, \*\*kwargs)\

def **transform** (self, inputs)\

## 额外继承的成员函数

# 详细描述

transform for multi hop features and label

This is inputs for Supervised graphSAGE

**Examples:**

>     >>> from galileo.pytorch import MultiHopFeatureLabelTransform
>     >>> transform = MultiHopFeatureLabelTransform([[0],[0]],[2,3],'label',7,
>             False,['feature'],5).transform
>     >>> res = transform([2,4])
>     >>> res.keys()
>     dict_keys(['features', 'labels'])
>     >>> res['labels'].shape
>     torch.Size([2, 7])
>     >>> res['features'].keys()
>     dict_keys(['ids', 'dense'])
>     >>> res['features']['ids'].shape
>     torch.Size([2, 9])
>     >>> res['features']['dense'].shape
>     torch.Size([2, 9, 5])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform.\_\_init\_\_ ( self, list metapath, list fanouts, str label_name, int label_dim, bool edge_weight = `False`, list dense_feature_names = `None`, dense_feature_dims = `None`, list sparse_feature_names = `None`, sparse_feature_dims = `None`, \*\* kwargs)

**参数**

> *metapath* list of list, edge types of multi hop\
> *fanouts* number of multi hop\
> *label_name* label feature name\
> *label_dim* label dim\
> *edge_weight* has weight or not\
> *dense_feature_names* list of str\
> *dense_feature_dims* int or list\[int\]\
> *sparse_feature_names* list of str\
> *sparse_feature_dims* int or list\[int\]

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.multi_hop_feature_label.MultiHopFeatureLabelTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(features=dict,labels=dict)\
> inner dict: dict(ids=tensor, dense=tensor, sparse=tensor,
> edge_weight=tensor)

重载
**galileo.framework.pytorch.python.transforms.multi_hop_feature.MultiHopFeatureTransform**
.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
