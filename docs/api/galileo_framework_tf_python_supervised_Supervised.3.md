---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.supervised.Supervised
---

# NAME

galileo.framework.tf.python.supervised.Supervised - supervised model

# SYNOPSIS

\

继承自 Model , 以及
**galileo.framework.python.base_supervised.BaseSupervised** .

## Public 成员函数

def **\_\_init\_\_** (self, loss_name=\_\_default_loss,
metric_names=\'categorical_accuracy\', dense_input_dim=None,
label_dim=None, num_classes=None, is_add_metrics=True, \*args,
\*\*kwargs)\

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

def **call** (self, inputs)\

def **get_config** (self)\

## Public 属性

**loss_name**\

**loss_obj**\

**metric_names**\

**is_add_metrics**\

**metric_objs**\

**dense_layer**\

**dense_input_dim**\

**label_dim**\

# 详细描述

supervised model

Methods that the subclass must implement: encoder

args:

**参数**

> *loss_name* loss name\
> *metric_names* metric names, default is categorical_accuracy\
> *dense_input_dim* input dim for dense layer\
> *label_dim* label dim\
> *num_classes* num of class\
> *is_add_metrics* add loss and metrics layers for keras

# 成员函数说明

## def galileo.framework.tf.python.supervised.Supervised.loss_and_metrics ( self, labels, logits)

**返回**

> a dict of loss and metrics or y_true and y_pred

重载 **galileo.framework.python.base_supervised.BaseSupervised** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
