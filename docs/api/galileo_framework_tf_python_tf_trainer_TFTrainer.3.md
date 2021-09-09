---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.tf.python.tf_trainer.TFTrainer
---

# NAME

galileo.framework.tf.python.tf_trainer.TFTrainer - Trainer for tf

# SYNOPSIS

\

继承自 **galileo.framework.python.base_trainer.BaseTrainer** .

被 **galileo.framework.tf.python.estimator_trainer.EstimatorTrainer** ,
以及 **galileo.framework.tf.python.keras_trainer.KerasTrainer** 继承.

## Public 成员函数

def **\_\_init\_\_** (self, model, inputs=None, module=None,
model_args=None, distribution_strategy=None, seed=None,
zk_server=DefaultValues.ZK_SERVER, zk_path=DefaultValues.ZK_PATH,
use_eager=False, soft_device_placement=True,
log_device_placement=False)\

def **get_dataset** (self, mode)\
get an dataset

def **do_train** (self)\

def **do_evaluate** (self)\

def **do_predict** (self)\

def **get_optimizer** (self)\
return an optimizer

def **train** (self, \*\*kwargs)\

def **evaluate** (self, \*\*kwargs)\

def **predict** (self, \*\*kwargs)\

def **init_model** (self, \*\*kwargs)\
init model

def **config_device** (self)\

def **parse_tf_config** (self)\

def **create_graph_client** (self)\

def **config_dist_strategy** (self)\

def **config_batch_num** (self)\

## Public 属性

**model_class**\

**model_args**\

**model_name**\

**should_dist_dataset**\

**run_config**\

**model_dir**\

**strategy**\

# 详细描述

Trainer for tf

**注解**

> use subclasses of **TFTrainer**, not this

**internel config**

> num_gpus task_type task_id num_chief num_workers num_ps is_chief

**internel attrs**

> run_config strategy model_name model_args optimizer latest_ckp
> estimator_config should_dist_dataset

# 构造及析构函数说明

## def galileo.framework.tf.python.tf_trainer.TFTrainer.\_\_init\_\_ ( self, model, inputs = `None`, module = `None`, model_args = `None`, distribution_strategy = `None`, seed = `None`, zk_server = `DefaultValues.ZK_SERVER`, zk_path = `DefaultValues.ZK_PATH`, use_eager = `False`, soft_device_placement = `True`, log_device_placement = `False`)

**参数**

> *model* instance or subclass of tf.keras.Model\
> *inputs* Inputs for model\
> *module* Module for trainer, use default\
> *model_args* args for model\
> *distribution_strategy* \'one_device\', \'mirrored\',
> \'multi_worker_mirrored\',\'parameter_server\'. default is None\
> *seed* seed for initializing training\
> *zk_server* zookeeper server address\
> *zk_path* zookeeper registration node name\
> *use_eager* bool, use eager when debug\
> *soft_device_placement* for tf.config.set_soft_device_placement\
> *log_device_placement* for tf.debugging.set_log_device_placement

# 成员函数说明

## def galileo.framework.tf.python.tf_trainer.TFTrainer.evaluate ( self, \*\* kwargs)

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

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

## def galileo.framework.tf.python.tf_trainer.TFTrainer.get_dataset ( self, mode)

get an dataset

**参数**

> *mode* train/evaluate/predict

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

被 **galileo.framework.tf.python.keras_trainer.KerasTrainer** 重载.

## def galileo.framework.tf.python.tf_trainer.TFTrainer.init_model ( self, \*\* kwargs)

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

被 **galileo.framework.tf.python.keras_trainer.KerasTrainer** , 以及
**galileo.framework.tf.python.estimator_trainer.EstimatorTrainer** 重载.

## def galileo.framework.tf.python.tf_trainer.TFTrainer.predict ( self, \*\* kwargs)

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

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

## def galileo.framework.tf.python.tf_trainer.TFTrainer.train ( self, \*\* kwargs)

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

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
