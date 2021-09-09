---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.estimator_trainer.EstimatorTrainer
---

# NAME

galileo.framework.tf.python.estimator_trainer.EstimatorTrainer - Trainer
for tf estimator

# SYNOPSIS

\

继承自 **galileo.framework.tf.python.tf_trainer.TFTrainer** .

## Public 成员函数

def **init_model** (self, \*\*kwargs)\
init model

def **create_estimator** (self)\

def **model_fn** (self, features, labels, mode)\

def **get_dataset** (self, mode, input_context=None)\

def **do_train** (self)\

def **do_evaluate** (self)\

def **do_predict** (self)\

## Public 属性

**estimator_config**\

**inputs_dict**\

**estimator**\

**model**\

# 详细描述

Trainer for tf estimator

attention API: galileo.tf.EstimatorTrainer

# 成员函数说明

## def galileo.framework.tf.python.estimator_trainer.EstimatorTrainer.init_model ( self, \*\* kwargs)

init model

**注解**

> all args are stored in run_config

**参数**

> *model_dir* model dir\
> *inputs_fn* inputs function, requried when self.inputs is None\
> *batch_size* Mini-batch size\
> *max_id* max vertex id\
> *batch_num* Number of mini-batch, default is \[max_id\] /
> \[batch_size\]\
> *log_steps* Number of steps to print log\
> *log_max_times_per_epoch* log max times per epoch, default is 100\
> *start_epoch* start of epoch\
> *num_epochs* number epochs\
> *optimizer* adam, sgd, momentum, adagrad\
> *learning_rate* learning rate\
> *momentum* momentum for optimizer\
> *save_checkpoint_epochs* The frequency to save checkpoint per epoch\
> *gpu_status* show gpu status\
> *save_predict_fn* callback for save results of predict
> save_predict_fn(ids, embeddings, dir, task_id)

**spacial args for tf**

**参数**

> *train_verbose*
>
> -   0 = silent
>
> -   1 = progress bar
>
> -   2 = one line per epoch
>
> \
> *tensorboard_steps* update tensorboard every steps\
> *profile_batch* int or pair\
> *estimator_hooks_fn* estimator hooks function for early stop
> (tf.estimator.experimental.stop_if_no_decrease_hook)\
> *callbacks* custom keras callbacks\
> *hooks* custom estimator hooks\
> *custom_metric_fn* custom estimator metric function\
> *eval_exporters* instance of tf.estimator.Exporter\
> *eval_hooks* for tf.estimator.EvalSpec\
> *eval_throttle_secs* throttle_secs arg for tf.estimator.EvalSpec\
> *keep_checkpoint_max* args for tf.estimator.RunConfig

**other args for input_fn**

重载 **galileo.framework.tf.python.tf_trainer.TFTrainer** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
