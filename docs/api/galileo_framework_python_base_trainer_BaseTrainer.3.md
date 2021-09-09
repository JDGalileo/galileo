---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.python.base_trainer.BaseTrainer
---

# NAME

galileo.framework.python.base_trainer.BaseTrainer - **BaseTrainer** for
tf and pytorch and more

# SYNOPSIS

\

继承自 metaclass , 以及 ABCMeta .

被 **galileo.framework.pytorch.python.trainer.Trainer** , 以及
**galileo.framework.tf.python.tf_trainer.TFTrainer** 继承.

## Public 成员函数

def **\_\_init\_\_** (self, model, **BaseInputs** inputs=None,
**BaseModule** module=None, dict **config**=None)\

def **config** (self)\
get config

def **get_dataset** (self, mode)\
get an dataset

def **get_optimizer** (self)\
return an optimizer

def **train** (self, \*\*kwargs)\
train

def **evaluate** (self, \*\*kwargs)\
evaluate

def **predict** (self, \*\*kwargs)\
predict

## Public 属性

**model**\

**inputs**\

**module**\

# 详细描述

**BaseTrainer** for tf and pytorch and more

-   setup distributed training

-   train/evaluate/predict

**注意**

> API: galileo.BaseTrainer

# 构造及析构函数说明

## def galileo.framework.python.base_trainer.BaseTrainer.\_\_init\_\_ ( self, model, **BaseInputs** inputs = `None`, **BaseModule** module = `None`, dict config = `None`)

**参数**

> *model* Model for tf or pytorch\
> *inputs* Inputs for model\
> *module* Module for trainer, use default\
> *config* dict, config

# 成员函数说明

## def galileo.framework.python.base_trainer.BaseTrainer.evaluate ( self, \*\* kwargs)

evaluate

**参数**

> *kwargs* config

被 **galileo.framework.tf.python.tf_trainer.TFTrainer** , 以及
**galileo.framework.pytorch.python.trainer.Trainer** 重载.

## def galileo.framework.python.base_trainer.BaseTrainer.get_dataset ( self, mode)

get an dataset

**参数**

> *mode* train/evaluate/predict

被 **galileo.framework.tf.python.tf_trainer.TFTrainer**,
**galileo.framework.tf.python.keras_trainer.KerasTrainer** , 以及
**galileo.framework.pytorch.python.trainer.Trainer** 重载.

## def galileo.framework.python.base_trainer.BaseTrainer.predict ( self, \*\* kwargs)

predict

**参数**

> *kwargs* config

被 **galileo.framework.tf.python.tf_trainer.TFTrainer** , 以及
**galileo.framework.pytorch.python.trainer.Trainer** 重载.

## def galileo.framework.python.base_trainer.BaseTrainer.train ( self, \*\* kwargs)

train

**参数**

> *kwargs* config

被 **galileo.framework.tf.python.tf_trainer.TFTrainer** , 以及
**galileo.framework.pytorch.python.trainer.Trainer** 重载.

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
