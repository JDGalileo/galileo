---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.transforms.neighbor_neg.NeighborNegTransform
---

# NAME

galileo.framework.pytorch.python.transforms.neighbor_neg.NeighborNegTransform
- neighbor with negative sampling

# SYNOPSIS

\

继承自 **galileo.framework.python.base_transform.BaseTransform** .

## Public 成员函数

def **\_\_init\_\_** (self, list vertex_type, list edge_types, int
negative_num, \*\*kwargs)\

def **transform** (self, inputs)\

# 详细描述

neighbor with negative sampling

# 构造及析构函数说明

## def galileo.framework.pytorch.python.transforms.neighbor_neg.NeighborNegTransform.\_\_init\_\_ ( self, list vertex_type, list edge_types, int negative_num, \*\* kwargs)

**参数**

> *vertex_type*\
> *edge_types*\
> *negative_num*

# 成员函数说明

## def galileo.framework.pytorch.python.transforms.neighbor_neg.NeighborNegTransform.transform ( self, inputs)

**参数**

> *inputs* vertices

**返回**

> dict(target=tensor,context=tensor,negative=tensor)

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
