---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.unsupervised.Unsupervised
---

# NAME

galileo.framework.pytorch.python.unsupervised.Unsupervised -
unsupervised network embedding model

# SYNOPSIS

\

继承自 Module , 以及
**galileo.framework.python.base_unsupervised.BaseUnsupervised** .

## Public 成员函数

def **\_\_init\_\_** (self, loss_name=\'neg_cross_entropy\',
metric_names=\'mrr\', \*args, \*\*kwargs)\

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

def **forward** (self, inputs)\

## Public 属性

**loss_name**\

**metric_names**\

# 详细描述

unsupervised network embedding model

compute the loss and metrics

Methods that the subclass must implement:\
target_encoder, context_encoder,

# 成员函数说明

## def galileo.framework.pytorch.python.unsupervised.Unsupervised.loss_and_metrics ( self, logits, negative_logits)

**返回**

> a dict of loss and metrics

重载 **galileo.framework.python.base_unsupervised.BaseUnsupervised** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
