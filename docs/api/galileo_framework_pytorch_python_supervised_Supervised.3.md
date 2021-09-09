---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.supervised.Supervised
---

# NAME

galileo.framework.pytorch.python.supervised.Supervised - supervised
model

# SYNOPSIS

\

继承自 Module , 以及
**galileo.framework.python.base_supervised.BaseSupervised** .

## Public 成员函数

def **\_\_init\_\_** (self, loss_name=\'multi_label_sm\',
metric_names=\'f1_score\', dense_input_dim=None, label_dim=None,
num_classes=None, \*args, \*\*kwargs)\

def **encoder** (self, inputs)\
supervised feature encoder

def **dense_encoder** (self, inputs)\
a dense layer after encoder

def **loss_and_metrics** (self, labels, logits)\

def **convert_ids_tensor** (self, inputs)\
convert ids tensor

def **convert_features_tensor** (self, inputs)\
convert features tensor

def **convert_labels_tensor** (self, inputs)\
convert labels tensor

def **forward** (self, inputs)\

## Public 属性

**loss_name**\

**metric_names**\

**dense_layer**\

**label_dim**\

# 详细描述

supervised model

compute the loss and metrics

Methods that the subclass must implement:\
encoder

**参数**

> *loss_name* loss name\
> *metric_names* metric names, default is f1_score\
> *dense_input_dim* input dim for dense layer\
> *label_dim* label dim\
> *num_classes* num of class

# 成员函数说明

## def galileo.framework.pytorch.python.supervised.Supervised.loss_and_metrics ( self, labels, logits)

**返回**

> a dict of loss and metrics

重载 **galileo.framework.python.base_supervised.BaseSupervised** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
