---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.transforms.relation.RelationTransform
---

# NAME

galileo.framework.tf.python.transforms.relation.RelationTransform -
transform multi hops to relation graph

# SYNOPSIS

\

继承自 **galileo.framework.python.base_transform.BaseTransform** .

## Public 成员函数

def **\_\_init\_\_** (self, list fanouts, bool sort_indices=False,
\*\*kwargs)\

def **transform** (self, inputs)\

## Public 属性

**fanouts**\

**fanouts_list**\

**fanouts_dim**\

**fanouts_indices**\

**sort_indices**\

# 详细描述

transform multi hops to relation graph

a relation graph is a dict:

    dict(
        relation_indices=tensor,
        relation_weight=tensor,
        target_indices=tensor,
    )

relation_indices is a \[2,E\] int tensor, E is number of edges,\
indices of relation/edge of graph relation_weight is a \[E,1\] float
tensor, weight of relation\
target_indices is indices of target vertices, \[batch size\]

**Examples**

>     >>> from galileo.tf import RelationTransform
>     >>> # fanouts= [2,3] batch size=5 num nodes=10
>     >>> ids = tf.random.uniform([5, 9], maxval=10, dtype=tf.int32)
>     >>> ids, indices = tf.unique(tf.reshape(ids, [-1]))
>     >>> rt = RelationTransform([2,3])
>     >>> res = rt.transform(dict(indices=indices,
>                 edge_weight=tf.random.normal((5,9))))
>     >>> res.keys()
>     dict_keys([relation_indices', 'relation_weight', 'target_indices'])
>     >>> res['relation_indices'].shape
>     TensorShape([2, 40])
>     >>> res['relation_weight'].shape
>     TensorShape([40, 1])
>     >>> res['target_indices'].shape
>     TensorShape([5])

# 构造及析构函数说明

## def galileo.framework.tf.python.transforms.relation.RelationTransform.\_\_init\_\_ ( self, list fanouts, bool sort_indices = `False`, \*\* kwargs)

**参数**

> *fanouts* number of multi hop\
> *sort_indices* sort relation indices

# 成员函数说明

## def galileo.framework.tf.python.transforms.relation.RelationTransform.transform ( self, inputs)

**参数**

> *inputs* list or tuple or\
> dict(indices=tensor, edge_weight=tensor)\
> size of indices and edge_weight must be N \* fanouts_dim

**返回**

> dict( relation_indices=tensor, relation_weight=tensor,
> target_indices=tensor, )

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
