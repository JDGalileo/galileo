# Galileo分布式训练

Galileo提供的[图模型示例](../examples/README.md)包含有TensorFlow和PyTorch版本的图算法模型实现，下面选择TensorFlow的GraphSAGE和Pytorch的Node2vec为例，分别介绍如何使用Galileo进行单机和分布式的图算法模型训练。

# 单机
1.在一个容器中启动zk服务，执行以下shell脚本启动

```bash
cd <galileo root>
/bin/bash examples/start_zk.sh
```

2.模型训练，评估，预测

例子会先训练，然后使用测试集进行评估模型，最后预测并保存顶点的embedding

TensorFlow示例：
```bash
python3 examples/tf/sage/supervised.py
```

PyTorch示例：
```bash
python3 examples/pytorch/node2vec/simple.py
```

# 分布式
Galileo支持TensorFlow和PyTorch的分布式训练。

Galileo提供了[docker镜像](https://hub.docker.com/r/jdgalileo/galileo)运行环境，分布式训练建议使用k8s进行部署。

接下来分别介绍如何进行TensorFlow和PyTorch分布式训练。

## 1. 图引擎服务
### zk服务
运行一个pod来启动zk服务，假设pod的地址是pod1，那么zk_server=pod1:2181
```bash
/bin/bash examples/start_zk.sh
```
### 单机图引擎服务
使用一个pod来启动一个图引擎服务。下面以cora数据集为例，指定zk_server为pod1:2181
```bash
galileo_service --data_source_name cora --role engine --zk_server=pod1:2181
```

### 分布式图引擎服务
为了应对工业级大数据量场景，图引擎支持分布式部署。

需要启动数个图引擎服务的pod，假设启动三个图引擎服务，以下分别是启动命令
```bash
galileo_service --data_source_name cora --role engine --zk_server=pod1:2181 --shard_index 0 --shard_num 3
galileo_service --data_source_name cora --role engine --zk_server=pod1:2181 --shard_index 1 --shard_num 3
galileo_service --data_source_name cora --role engine --zk_server=pod1:2181 --shard_index 2 --shard_num 3
```

## 2. TensorFlow
TensorFlow版本支持两种分布式训练模式：AllReduce模式和Parameter Server模式

tf的任务可以使用[tf operator](https://github.com/kubeflow/tf-operator)来完成，就不需要手动配置TF_CONFIG了。

### 2.1 AllReduce模式

假设启动一个2worker训练任务，首先需要配置好TF_CONFIG环境变量：
```
pod2：
export TF_CONFIG='{ "cluster": { "worker": ["pod2:2223", "pod3:2224"] }, "task": {"type": "worker", "index": 0} }'

pod3：
export TF_CONFIG='{ "cluster": { "worker": ["pod2:2223", "pod3:2224"] }, "task": {"type": "worker", "index": 1} }'
```
分别在pod2和pod3上启动训练：
```
pod2：
python3 examples/tf/sage/supervised.py --role worker --zk_server=pod1:2181 --ds multi_worker_mirrored

pod3：
python3 examples/tf/sage/supervised.py --role worker --zk_server=pod1:2181 --ds multi_worker_mirrored

```

### 2.2 Parameter Server

假设启动一个1ps + 1chief + 1worker的训练任务， 首先需要配置好TF_CONFIG环境变量：
```
pod2：
export TF_CONFIG='{ "cluster": {"ps": ["pod2:2222"], "chief": ["pod3:2223"], "worker": ["pod4:2224"] }, "task": {"type": "ps", "index": 0} }'

pod3：
export TF_CONFIG='{ "cluster": {"ps": ["pod2:2222"], "chief": ["pod3:2223"], "worker": ["pod4:2224"] }, "task": {"type": "chief", "index": 0} }'

pod4：
export TF_CONFIG='{ "cluster": {"ps": ["pod2:2222"], "chief": ["pod3:2223"], "worker": ["pod4:2224"] }, "task": {"type": "worker", "index": 0} }'
```

在pod2、pod3、pod4上分别启动训练服务：
```

pod2上启动parameter server服务：
python3 examples/tf/sage/supervised.py --role ps --zk_server=pod1:2181 --ds parameter_server

pod3上启动训练任务：
python3 examples/tf/sage/supervised.py --role worker --zk_server=pod1:2181 --ds parameter_server

pod4上启动训练任务：
python3 examples/tf/sage/supervised.py --role worker --zk_server=pod1:2181 --ds parameter_server
```

## 3. PyTorch

PyTorch版本仅支持AllReduce模式

假设启动一个2worker训练任务，首先配置环境变量：
```
pod2机器：
export MASTER_ADDR=pod2
export MASTER_PORT=2222
export WORLD_SIZE=2
export RANK=0

pod3机器：
export MASTER_ADDR=pod2
export MASTER_PORT=2222
export WORLD_SIZE=2
export RANK=1
```

在pod2和pod3上分别启动训练任务：
```
pod2上启动训练任务：
python3 examples/pytorch/node2vec/node2vec_simple.py --role worker --zk_server=pod1:2181

pod3上启动训练任务：
python3 examples/pytorch/node2vec/node2vec_simple.py --role worker --zk_server=pod1:2181
```
