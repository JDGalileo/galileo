# Galileo图模型例子列表

| 模型 | 框架 | 框架 |
| --------- | -------------- | ------- |
| LINE      | [Tensorflow简单用法](tf/line/simple.py)（keras和estimator）<br>[Tensorflow高级用法](tf/line/advance.py)（keras） | [PyTorch](pytorch/line/line.py) |
| Node2vec  |[Tensorflow简单用法](tf/node2vec/simple.py)（keras和estimator）<br>[Tensorflow使用hdfs文件](tf/node2vec/simple_hdfs.py)（keras和estimator）<br>[Tensorflow使用一个ps训练](tf/node2vec/simple_ps.py)（estimator）<br>[Tensorflow使用多个ps训练](tf/node2vec/multi_ps.py)（estimator）| [PyTorch简单用法](pytorch/node2vec/simple.py) <br>[PyTorch原生训练](pytorch/node2vec/pytorch.py) <br>[PyTorch高级用法](pytorch/node2vec/advance.py) |
| 有监督GraphSAGE | [Tensorflow](tf/sage/supervised.py)（keras） | [PyTorch](pytorch/sage/supervised.py) |
| 无监督GraphSAGE | [Tensorflow](tf/sage/unsupervised.py)（keras）<br>[Tensorflow](tf/sage/unsupervised_estimator.py)（estimator）<br>[Tensorflow sparse模型](tf/sage/unsupervised_sparse.py)（estimator）<br>[Tensorflow自定义predict](tf/sage/unsupervised_custom_predict.py)（estimator）| [PyTorch](pytorch/sage/unsupervised.py) <br>[PyTorch sparse模型](pytorch/sage/unsupervised_sparse.py) <br>[PyTorch使用hdfs文件](pytorch/sage/unsupervised_hdfs.py)|
| [GCN](tf/gcn/README.md) | [Tensorflow](tf/gcn/gcn.py)（estimator） | - |
| [GATNE直推式](tf/GATNE/README.md) | [Tensorflow](tf/GATNE/transductive.py)（estimator） | - |
| [GATNE归纳式](tf/GATNE/README.md) | [Tensorflow](tf/GATNE/inductive.py)（estimator）<br/>[Tensorflow支持多种特征](tf/GATNE/custom_inductive.py)（estimator） | - |
| [HeteSAGE自研模型](tf/heteSAGE/README.md) | [Tensorflow](tf/heteSAGE/semi.py)（estimator） | - |
| [HAN](tf/HAN/README.md) | [Tensorflow](tf/HAN/supervised.py)（estimator） | - |

备注：
1. 简单用法是指使用Galileo提供的Layer实现模型，高级用法是指自定义模型实现。
1. Tensorflow后端下所有模型都支持keras和estimator训练，只是有的模型只提供了部分实现，另外如果使用ps的话只能使用estimator训练了。
1. Unify统一框架，使用参数选择使用Tensorflow或PyTorch，目前某些API还不太完善，只提供了[LINE模型](unify/line.py)的示例，最好是使用各自框架的例子。

# 运行例子
1. 在docker中启动zookeeper
```
bash start_zk.sh
```
2. 运行例子
```
python3 <例子路径> <可选参数>
```
