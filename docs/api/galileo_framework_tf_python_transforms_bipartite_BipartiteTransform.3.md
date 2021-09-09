---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.transforms.bipartite.BipartiteTransform
---

# NAME

galileo.framework.tf.python.transforms.bipartite.BipartiteTransform -
transform to convert bipartites

# SYNOPSIS

\

继承自 **galileo.framework.python.base_transform.BaseTransform** .

## Public 成员函数

def **\_\_init\_\_** (self, list fanouts, \*\*kwargs)\

def **transform** (self, inputs)\

## Public 属性

**fanouts**\

**fanouts_list**\

# 详细描述

transform to convert bipartites

a bipartite is a dict:

    dict(
        src=tensor,
        dst=tensor,
        src_feature=tensor,
        dst_feature=tensor,
        edge_weight=tensor,
    )

**Examples**

>     >>> from galileo.tf import BipartiteTransform
>     >>> bt = BipartiteTransform([2,3])
>     >>> ids = tf.random.uniform((4,9), maxval=10, dtype=tf.int32)
>     >>> res = bt.transform(dict(ids=ids, feature=tf.random.normal((4,9,16)),
>                 edge_weight=tf.random.normal((4,9))))
>     >>> len(res)
>     2
>     >>> res[0]['src'].shape
>     TensorShape([4, 2])
>     >>> res[0]['dst'].shape
>     TensorShape([4, 6])
>     >>> res[0]['src_feature'].shape
>     TensorShape([4, 2, 16])
>     >>> res[0]['dst_feature'].shape
>     TensorShape([4, 6, 16])
>     >>> res[0]['edge_weight'].shape
>     TensorShape([4, 6])
>     >>> res[1]['src'].shape
>     TensorShape([4, 1])
>     >>> res[1]['dst'].shape
>     TensorShape([4, 2])
>     >>> res[1]['src_feature'].shape
>     TensorShape([4, 1, 16])
>     >>> res[1]['dst_feature'].shape
>     TensorShape([4, 2, 16])
>     >>> res[1]['edge_weight'].shape
>     TensorShape([4, 2])

# 构造及析构函数说明

## def galileo.framework.tf.python.transforms.bipartite.BipartiteTransform.\_\_init\_\_ ( self, list fanouts, \*\* kwargs)

**参数**

> *fanouts* number of multi hop

# 成员函数说明

## def galileo.framework.tf.python.transforms.bipartite.BipartiteTransform.transform ( self, inputs)

**参数**

> *inputs* dict(ids=tensor,feature=tensor,edge_weight=tensor)

**返回**

> list of bipartite\
> items in bipartites are arranged in the direction of aggregation, one
> of item:\
>
> -   src -\> dst are direction of edges
>
> -   dst -\> src are direction of aggregation
>
> -   src shape: (\*, fanouts_list\[i-1\])
>
> -   dst edge_weight shape: (\*, fanouts_list\[i\])
>
> may have duplicated vertices in src and dst

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
