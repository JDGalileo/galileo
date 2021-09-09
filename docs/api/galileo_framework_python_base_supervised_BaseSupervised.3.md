---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_supervised.BaseSupervised
---

# NAME

galileo.framework.python.base_supervised.BaseSupervised - base
supervised model for graph

# SYNOPSIS

\

继承自 metaclass , 以及 ABCMeta .

被 **galileo.framework.pytorch.python.supervised.Supervised** , 以及
**galileo.framework.tf.python.supervised.Supervised** 继承.

## Public 成员函数

def **\_\_init\_\_** (self, label_dim=None, num_classes=None, \*args,
\*\*kwargs)\

def **encoder** (self, inputs)\
supervised feature encoder

def **dense_encoder** (self, inputs)\
a dense layer after encoder

def **loss_and_metrics** (self, labels, logits)\
compute loss and metrics

def **convert_ids_tensor** (self, inputs)\
convert ids tensor

def **convert_features_tensor** (self, inputs)\
convert features tensor

def **convert_labels_tensor** (self, inputs)\
convert labels tensor

def **unpack_sample** (self, inputs)\
unpack sample

def **\_\_call\_\_** (self, inputs, \*\*kwargs)\

## Public 属性

**label_dim**\

**num_classes**\

# 详细描述

base supervised model for graph

**参数**

> *label_dim* label dim\
> *num_classes* num of class

# 成员函数说明

## def galileo.framework.python.base_supervised.BaseSupervised.\_\_call\_\_ ( self, inputs, \*\* kwargs)

**参数**

> *inputs* dict of tensors\
> contains target, features and labels

**参见**

> **unpack_sample**

## def galileo.framework.python.base_supervised.BaseSupervised.unpack_sample ( self, inputs)

unpack sample

**参数**

> *inputs* dict, keys of dict:\
> case 1: (features, target) for save embedding\
> case 2: (features, labels) for train and evaluate

**返回**

> features, target_or_labels, only_embedding

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
