---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.rw_neg.RandomWalkNegTransform
---

# NAME

galileo.framework.pytorch.python.transforms.rw_neg.RandomWalkNegTransform
- randomwalk with negative sampling

# SYNOPSIS

\

继承自 **galileo.framework.python.base_transform.BaseTransform** .

## Public 成员函数

def **\_\_init\_\_** (self, list vertex_type, list edge_types, int
negative_num, int context_size, int repetition=1, float walk_p=1., float
walk_q=1., walk_length=None, metapath=None, \*\*kwargs)\

def **transform** (self, inputs)\

# 详细描述

randomwalk with negative sampling

**example**

>     >>> from galileo.pytorch import RandomWalkNegTransform
>     >>> transform = RandomWalkNegTransform([0],[0],3,2,1,
>     ... walk_length=3).transform
>     >>> res = transform([2,4,6])
>     >>> res.keys()
>     dict_keys(['target', 'context', 'negative'])
>     >>> res['target'].shape
>     torch.Size([30, 1])
>     >>> res['context'].shape
>     torch.Size([30, 1])
>     >>> res['negative'].shape
>     torch.Size([30, 3])

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.rw_neg.RandomWalkNegTransform.\_\_init\_\_ ( self, list vertex_type, list edge_types, int negative_num, int context_size, int repetition = `1`, float walk_p = `1.`, float walk_q = `1.`, walk_length = `None`, metapath = `None`, \*\* kwargs)

**参数**

> *vertex_type*\
> *edge_types*\
> *negative_num*\
> *context_size*\
> *repetition*\
> *walk_p*\
> *walk_q*\
> *walk_length*\
> *metapath*

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.rw_neg.RandomWalkNegTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(target=tensor,context=tensor,negative=tensor)

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
