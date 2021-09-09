# Galileo入门教程

本教程将使用Galileo应用GNN在图顶点分类任务中。

通过本教程，你将学会：

1. 启动Galileo图引擎服务，加载图数据

2. 使用Galileo提供的Transform构建模型输入

3. 使用Galileo提供的Layer构建图模型

4. 训练评估和预测模型

## GraphSAGE模型

本教程使用有监督的GraphSAGE模型做图顶点分类任务。GraphSAGE模型通过聚合邻居顶点的特征来更新目标顶点来学习一种顶点的表示，再结合顶点的标签，更新模型的参数，通过多轮的训练收敛后得到最终的图模型来预测未见过的顶点标签。

本教程使用tensorflow作为后端。完整的代码见[sage/supervised.py](../examples/tf/sage/supervised.py)。

## 1.导入包

```python
import os
import argparse
import tensorflow as tf
import galileo as g
import galileo.tf as gt
```

## 启动Galileo图引擎服务

本教程使用Cora图数据集，Galileo框架集成了Cora数据集，会自动下载并转换为Galileo图引擎服务需要的格式。除了cora数据集，Galileo还集成了ppi, citeseer, pubmed。准备自己的图数据集可以参考[图数据准备](data_prepare.md)。

以下演示了启动Galileo图引擎服务，并使用Cora图数据集的过程。

```python
parser = g.define_service_args()
args, _ = parser.parse_known_args()
if args.data_source_name is None:
  args.data_source_name = 'cora'   #如果没有传入data_source_name使用cora作为默认
g.start_service_from_args(args)
```

`galileo.start_service_from_args`函数会根据data_source_name自动下载和转换为Galileo图引擎服务需要的格式，并启动一个后台的Galileo图引擎服务。更多的参数说明见[API](api.md)。

注意：Galileo图引擎服务依赖zookeeper，启动Galileo图引擎服务前要先[启动zookeeper](../examples/README.md#运行例子)。

## 构建模型输入

继承`galileo.BaseInputs`实现其中的`train_data`, `evaluate_data`, `predict_data`方法，分别得到用来训练，评估和预测的dataset。

训练和评估使用`gt.MultiHopFeatureLabelTransform`处理输入的顶点，得到多阶邻居的特征和标签。不同的是训练在图中随机采样顶点（`gt.VertexDataset`），评估是从测试样本中获取顶点（`g.get_test_vertex_ids`）。预测则使用使用所有顶点。

更多的参数说明和Transforms见[API](api.md#galileobasetransform)。

```python
class Inputs(g.BaseInputs):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)
        # 使用galileo内置的Transform，例子中使用多跳的邻居采样+标签的Transform
        self.transform = gt.MultiHopFeatureLabelTransform(
            **self.config).transform

    # 定义train的数据，从图服务中采样顶点
    def train_data(self):
        return gt.dataset_pipeline(gt.VertexDataset, self.transform,
                                   **self.config)

    def evaluate_data(self):
        # 使用get_test_vertex_ids接口得到data_source_name的测试集
        test_ids = g.get_test_vertex_ids(
            data_source_name=self.config['data_source_name'])
        return gt.dataset_pipeline(
            lambda **kwargs: gt.TensorDataset(test_ids, **kwargs),
            self.transform, **self.config)

    # 定义predict的数据，使用全部顶点集合
    def predict_data(self):
        def predict_transform(inputs):
            outputs = self.transform(inputs)
            outputs['target'] = inputs
            return outputs

        return gt.dataset_pipeline(
            lambda **kwargs: gt.RangeDataset(
                start=0, end=kwargs['max_id'], **kwargs), predict_transform,
            **self.config)
```

## 构建图模型

本教程构建的有监督的GraphSAGE模型继承自`gt.Supervised`，其实现了有监督模型的一般流程。

以下实现了两层的GraphSAGE模型，使用Galileo提供的`gt.SAGELayer`实现了核心的graphSAGE的message passing消息传递聚合和更新的逻辑。

更多的参数说明见[API](api.md)。

```python
class SupSAGE(gt.Supervised):
    def __init__(self,
                 hidden_dim,
                 num_classes,
                 dense_feature_dims,
                 fanouts,
                 aggregator_name='mean',
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_combiner = gt.FeatureCombiner(
            dense_feature_dims=dense_feature_dims)
        self.layer0 = gt.SAGELayer(hidden_dim,
                                   aggregator_name,
                                   activation='relu',
                                   dropout_rate=dropout_rate)
        self.layer1 = gt.SAGELayer(num_classes,
                                   aggregator_name,
                                   dropout_rate=dropout_rate)
        self.to_bipartite = gt.BipartiteTransform(fanouts).transform

    def encoder(self, inputs):
        feature = self.feature_combiner(inputs)
        bipartites = self.to_bipartite(dict(feature=feature))
        bipartites = self.layer0(bipartites)
        bipartites = self.layer1(bipartites)
        output = bipartites[-1]['src_feature']
        output = tf.squeeze(output)
        return output
```

## 训练评估和预测模型

```python
trainer = gt.KerasTrainer(SupSAGE, inputs, model_args=model_args)
trainer.train(**model_config)
trainer.evaluate(**model_config)
trainer.predict(**model_config)
```

更多的参数说明见[API](api.md)。

PyTorch后端的例子和Tensorflow后端类似，Galileo已经将接口统一了。可以参考[examples中的例子](../examples/README.md)。

