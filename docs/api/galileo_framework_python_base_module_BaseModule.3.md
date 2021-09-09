---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_module.BaseModule
---

# NAME

galileo.framework.python.base_module.BaseModule - Base Module for tf and
pytorch and more

# SYNOPSIS

\

继承自 **galileo.framework.python.named_object.NamedObject**, metaclass
, 以及 ABCMeta .

## Public 成员函数

def **\_\_init\_\_** (self, dict **config**=None, str name=None)\

def **config** (self)\
get config

def **train_step** (self, inputs, model, optimizer)\
train step, including forward and backward

def **evaluate_step** (self, inputs, model)\
evaluate step

def **predict_step** (self, inputs, model)\
predict step

# 详细描述

Base Module for tf and pytorch and more

**参数**

> *config* config dict for module\
> *name* module name

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
