---
date: 2021年 九月 1日 星期三
section: 3
title: galileo.framework.pytorch.python.trainer.Trainer
---

# NAME

galileo.framework.pytorch.python.trainer.Trainer - pytorch trainer

# SYNOPSIS

\

继承自 **galileo.framework.python.base_trainer.BaseTrainer** .

## Public 成员函数

def **\_\_init\_\_** (self, model, inputs=None, module=None,
multiprocessing_distributed=False, num_procs_per_shard=None, rank=0,
world_size=1, dist_url=None, dist_backend=None, seed=None,
zk_server=DefaultValues.ZK_SERVER, zk_path=DefaultValues.ZK_PATH)\

def **get_optimizer** (self)\
return an optimizer

def **train** (self, \*\*kwargs)\

def **evaluate** (self, \*\*kwargs)\

def **predict** (self, \*\*kwargs)\

def **run** (self, \*\*kwargs)\
run train, evaluate, predict

def **run_worker** (self, local_rank)\

def **create_client** (self)\

def **get_dataset** (self, mode)\
get an dataset

def **do_train** (self)\

def **do_evaluate** (self)\

def **do_predict** (self)\

## Public 属性

**module**\

**run_config**\

**model**\

# 详细描述

pytorch trainer

**internal attrs:**

> use_cuda global_rank local_rank is_master

**注意**

> API: galileo.pytoch.Trainer

# 构造及析构函数说明

## def galileo.framework.pytorch.python.trainer.Trainer.\_\_init\_\_ ( self, model, inputs = `None`, module = `None`, multiprocessing_distributed = `False`, num_procs_per_shard = `None`, rank = `0`, world_size = `1`, dist_url = `None`, dist_backend = `None`, seed = `None`, zk_server = `DefaultValues.ZK_SERVER`, zk_path = `DefaultValues.ZK_PATH`)

**参数**

> *model* instance of torch.nn.Module\
> *inputs* Inputs for model\
> *module* Module for trainer, use default\
> *multiprocessing_distributed* multi-processing distributed training\
> *num_procs_per_shard* use default\
> *rank* read from env RANK, use default\
> *world_size* read from env WORLD_SIZE, use default\
> *dist_url* use default\
> *dist_backend* use default\
> *seed* seed for initializing training\
> *zk_server* zookeeper server address\
> *zk_path* zookeeper registration node name

# 成员函数说明

## def galileo.framework.pytorch.python.trainer.Trainer.evaluate ( self, \*\* kwargs)

**参数**

> *model_dir* model dir\
> *inputs_fn* inputs function, requried when self.inputs is None\
> *log_steps* Number of steps to print log\
> *log_max_times_per_epoch* log max times per epoch, default is 100\
> *start_epoch* start of epoch\
> *num_epochs* number epochs\
> *optimizer* adam, sgd, momentum\
> *learning_rate* learning rate\
> *momentum* momentum for optimizer\
> *save_checkpoint_epochs* The frequency to save checkpoint per epoch\
> *gpu_status* bool show gpu status\
> *save_predict_fn* callback for save results of predict
> save_predict_fn(ids, embeddings, dir, rank)\
> *save_best_model* bool, save the best model

**spacial params for pytorch**

**参数**

> *weight_decay* weight_decay for optimizer\
> *resume* file to checkpoint\
> *hooks* hooks for log metrics and so on

**params for inputs_fn**

**参数**

> *batch_size* Mini-batch size\
> *max_id* max vertex id\
> *batch_num* Number of mini-batch, default is \[max_id\] /
> \[batch_size\]

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

## def galileo.framework.pytorch.python.trainer.Trainer.get_dataset ( self, mode)

get an dataset

**参数**

> *mode* train/evaluate/predict

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

## def galileo.framework.pytorch.python.trainer.Trainer.predict ( self, \*\* kwargs)

**参数**

> *model_dir* model dir\
> *inputs_fn* inputs function, requried when self.inputs is None\
> *log_steps* Number of steps to print log\
> *log_max_times_per_epoch* log max times per epoch, default is 100\
> *start_epoch* start of epoch\
> *num_epochs* number epochs\
> *optimizer* adam, sgd, momentum\
> *learning_rate* learning rate\
> *momentum* momentum for optimizer\
> *save_checkpoint_epochs* The frequency to save checkpoint per epoch\
> *gpu_status* bool show gpu status\
> *save_predict_fn* callback for save results of predict
> save_predict_fn(ids, embeddings, dir, rank)\
> *save_best_model* bool, save the best model

**spacial params for pytorch**

**参数**

> *weight_decay* weight_decay for optimizer\
> *resume* file to checkpoint\
> *hooks* hooks for log metrics and so on

**params for inputs_fn**

**参数**

> *batch_size* Mini-batch size\
> *max_id* max vertex id\
> *batch_num* Number of mini-batch, default is \[max_id\] /
> \[batch_size\]

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

## def galileo.framework.pytorch.python.trainer.Trainer.run ( self, \*\* kwargs)

run train, evaluate, predict

**参数**

> *mode* str, train, evaluate, predict\
> *model_dir* model dir\
> *inputs_fn* inputs function, requried when self.inputs is None\
> *log_steps* Number of steps to print log\
> *log_max_times_per_epoch* log max times per epoch, default is 100\
> *start_epoch* start of epoch\
> *num_epochs* number epochs\
> *optimizer* adam, sgd, momentum\
> *learning_rate* learning rate\
> *momentum* momentum for optimizer\
> *save_checkpoint_epochs* The frequency to save checkpoint per epoch\
> *gpu_status* bool show gpu status\
> *save_predict_fn* callback for save results of predict
> save_predict_fn(ids, embeddings, dir, rank)\
> *save_best_model* bool, save the best model

**spacial params for pytorch**

**参数**

> *weight_decay* weight_decay for optimizer\
> *resume* file to checkpoint\
> *hooks* hooks for log metrics and so on

**params for inputs_fn**

**参数**

> *batch_size* Mini-batch size\
> *max_id* max vertex id\
> *batch_num* Number of mini-batch, default is \[max_id\] /
> \[batch_size\]

## def galileo.framework.pytorch.python.trainer.Trainer.train ( self, \*\* kwargs)

**参数**

> *model_dir* model dir\
> *inputs_fn* inputs function, requried when self.inputs is None\
> *log_steps* Number of steps to print log\
> *log_max_times_per_epoch* log max times per epoch, default is 100\
> *start_epoch* start of epoch\
> *num_epochs* number epochs\
> *optimizer* adam, sgd, momentum\
> *learning_rate* learning rate\
> *momentum* momentum for optimizer\
> *save_checkpoint_epochs* The frequency to save checkpoint per epoch\
> *gpu_status* bool show gpu status\
> *save_predict_fn* callback for save results of predict
> save_predict_fn(ids, embeddings, dir, rank)\
> *save_best_model* bool, save the best model

**spacial params for pytorch**

**参数**

> *weight_decay* weight_decay for optimizer\
> *resume* file to checkpoint\
> *hooks* hooks for log metrics and so on

**params for inputs_fn**

**参数**

> *batch_size* Mini-batch size\
> *max_id* max vertex id\
> *batch_num* Number of mini-batch, default is \[max_id\] /
> \[batch_size\]

重载 **galileo.framework.python.base_trainer.BaseTrainer** .

# 作者

由 Doyxgen 通过分析 Galileo 的 源代码自动生成.
