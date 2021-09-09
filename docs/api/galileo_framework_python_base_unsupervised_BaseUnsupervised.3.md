---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_unsupervised.BaseUnsupervised
---

# NAME

galileo.framework.python.base_unsupervised.BaseUnsupervised - base
unsupervised model for graph

# SYNOPSIS

\

继承自 metaclass , 以及 ABCMeta .

被 **galileo.framework.pytorch.python.unsupervised.Unsupervised** , 以及
**galileo.framework.tf.python.unsupervised.Unsupervised** 继承.

## Public 成员函数

def **\_\_init\_\_** (self, \*args, \*\*kwargs)\

def **target_encoder** (self, inputs)\
unsupervised target encoder

def **context_encoder** (self, inputs)\
unsupervised context encoder

def **compute_logits** (self, target, context)\
compute logits

def **loss_and_metrics** (self, logits, negative_logits)\

def **convert_ids_tensor** (self, inputs)\
convert ids tensor

def **convert_features_tensor** (self, inputs)\
convert features tensor

def **unpack_sample** (self, inputs)\
unpack sample

def **\_\_call\_\_** (self, inputs, \*\*kwargs)\

# 详细描述

base unsupervised model for graph

including target embedding and context embedding

# 成员函数说明

## def galileo.framework.python.base_unsupervised.BaseUnsupervised.\_\_call\_\_ ( self, inputs, \*\* kwargs)

**参数**

> *inputs* dict of tensors,

**参见**

> **unpack_sample**

**返回**

> a dict of ids and embeddings if only_embedding is True, otherwise
> return a dict of loss and metrics

## def galileo.framework.python.base_unsupervised.BaseUnsupervised.loss_and_metrics ( self, logits, negative_logits)

**返回**

> a dict of loss and metrics

被 **galileo.framework.tf.python.unsupervised.Unsupervised** , 以及
**galileo.framework.pytorch.python.unsupervised.Unsupervised** 重载.

## def galileo.framework.python.base_unsupervised.BaseUnsupervised.unpack_sample ( self, inputs)

unpack sample

**参数**

> *inputs* dict, keys of dict:\
> case 1: (target,) for save embedding, target and target_ids are same\
> case 2: (target, target_ids) for save embedding, target is features of
> target_ids\
> case 3: (target, context, negative) for train and evaluate, all is
> features

**返回**

> target, target_ids, None, only_embedding=True\
> target, context, negative, only_embedding=False

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
