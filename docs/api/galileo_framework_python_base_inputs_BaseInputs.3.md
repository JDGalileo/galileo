---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_inputs.BaseInputs
---

# NAME

galileo.framework.python.base_inputs.BaseInputs - Base input

# SYNOPSIS

\

继承自 **galileo.framework.python.named_object.NamedObject**, metaclass
, 以及 ABCMeta .

## Public 成员函数

def **\_\_init\_\_** (self, dict **config**=None, str name=None)\

def **config** (self)\
get config

def **train_data** (self)\
get train data

def **evaluate_data** (self)\
get evaluate data

def **predict_data** (self)\
get predict data

# 详细描述

Base input

# 构造及析构函数说明

## def galileo.framework.python.base_inputs.BaseInputs.\_\_init\_\_ ( self, dict config = `None`, str name = `None`)

**参数**

> *config* dict, config\
> *name* name of inputs

# 成员函数说明

## def galileo.framework.python.base_inputs.BaseInputs.evaluate_data ( self)

get evaluate data

**返回**

> dataset

## def galileo.framework.python.base_inputs.BaseInputs.predict_data ( self)

get predict data

**返回**

> dataset

## def galileo.framework.python.base_inputs.BaseInputs.train_data ( self)

get train data

**返回**

> dataset

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
