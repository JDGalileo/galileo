# 图数据准备

Galileo框架已经集成了ppi, citeseer, pubmed, cora数据集。

如果想要训练自己的图数据，那么需要使用Galileo提供的图数据格式转换工具转换为Galileo图引擎服务需要的格式。

然后使用`data_path`指定路径，支持本地文件和HDFS文件，如果是HDFS文件需要同时指定hdfs_addr参数。

```
args.hdfs_addr = 'hdfs://'
args.data_path = '/path'
g.start_service_from_args(args)
```

## 图数据格式转换工具

## 工具介绍
- 基于Galileo设计的图源数据通用格式转换成图引擎所需要的二进制格式。
- 支持HDFS文件和本地文件

## 工具使用
- 工具名称：galileo_convertor

- 参数说明
  - --vertex_source_path 点图数据通用格式文件路径
  - --edge_source_path 边图数据通用格式文件路径
  - --schema_path 数据schema路径
  - --output_binary_path 点边输出的二进制格式文件路径
  - --partition_num 拆分文件分片个数，有利于分布式数据加载
  - --parallel 生成数据所用的线程数，理论上线程数越多，效率越高，但是考虑到磁盘io的效率，可以酌情配置
  - --hdfs_addr hdfs地址(此地址必须以`hdfs://`开头)，如果此地址设置了，那么都是从hadoop集群中读写文件, 如果不设置，那么都是在本地读取和写入
  - --hdfs_port hdfs端口，一般不用设置，默认0即可
  - --field_separator 源文件当中字段之间的分隔符，默认是"\t"
  - --array_separator 源文件中数组之间的分隔符，默认是","

- 工具单机docker中使用
    ```bash
    galileo_convertor --vertex_source_path testdata/vertex_source --edge_source_path testdata/edge_source --schema_path testdata/schema.json --output_binary_path testdata/binary --partition_num 2
    ```
    说明：为了减少图服务启动时的参数，我们约定schema_path和output_binary_path位于同一层目录下，schema_path的名称约定为schema.json，output_binary_path的目录名称为binary，这样约定后启动图服务数据目录只需要指定为testdata即可。

- 工具分布式使用
  针对大规模图数据无法在单机使用的问题，工具支持分布式运行

  注意：分布式运行工具的时候，需要保证图的顶点和边源数据文件个数要大于等于worker数。

  编写convertor.py

  ```python
  import galileo as g
  g.convert(vertex_source_path='/path/vertex',
            edge_source_path='/path/edge',
            schema_path='/path/schema.json',
            output_binary_path='/path/binary',
            hdfs_addr='hdfs://',
            partition_num=2,
            worker_index=0,
            worker_num=1)
  ```

​       然后在不同的worker上的docker中执行python convertor.py，需要配置对应的worker_index和worker_num。

​      也支持在k8s中运行，在k8s使用[tf operator](https://github.com/kubeflow/tf-operator)启动tfjob，convertor.py不再需要传入worker_index和worker_num参数，脚本会自动获取pod中的环境变量。

## schema文件介绍

- schema用途
定义vertex和edge的类型以及属性，方便直观了解图数据结构以及galileo服务器端加载、解析数据
- 支持的数据类型  
包括基本类型(DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,DT_UINT32,DT_INT64,DT_UINT64,DT_FLOAT,DT_DOUBLE)以及其对应的数组类型, 以及DT_STRING和DT_BINARY.
- 文件格式
```json
{
    "vertexes":
    [
        {
            "vtype": 0,
            "entity": "DT_INT64",
            "weight": "DT_FLOAT",
            "attrs":
            [
                { "name": "cid",   "dtype": "DT_INT16" },
                { "name": "price", "dtype": "DT_FLOAT" }
            ]
        },
        {
            "vtype": 1,
            "entity": "DT_INT64",
            "weight": "DT_FLOAT",
            "attrs":
            [
                { "name": "age",  "dtype": "DT_INT16" },
                { "name": "edu",  "dtype": "DT_ARRAY_INT64"},
                { "name": "addr", "dtype": "DT_FLOAT"},
                { "name": "test", "dtype": "DT_ARRAY_INT32" }
            ]
        }
    ],
    "edges":
    [
        {
            "etype": 0,
            "entity_1": "DT_INT64",
            "entity_2": "DT_INT64",
            "weight": "DT_FLOAT",
            "attrs": []
        },
        {
            "etype": 1,
            "entity_1": "DT_INT64",
            "entity_2": "DT_INT64",
            "weight": "DT_FLOAT",
            "attrs":
            [
                { "name": "discounts", "dtype": "DT_FLOAT" },
                { "name": "purchase_num", "dtype": "DT_INT32" },
                { "name": "test", "dtype": "DT_ARRAY_INT32" },
		        { "name": "attr1", "dtype": "DT_INT64"}
            ]
        }
    ]
}
```
说明：必须要包含vertexes和edges字段，vertexes中必须要有vtype,entity,weight,attrs，edges中必须要有etype,entity_1,entity_2,weight,attrs，所有属性都放在attrs中。

vtype和etype分别是顶点和边的类型，必须从0开始，依次递增 entity, entity_1和entity_2是顶点的id，全图唯一ID，

entity_1和entity_2分别是起始顶点和终止顶点 attrs定义顶点和边的属性集，

通过名称来指定属性

## 图源数据格式介绍
- 图数据格式
    schema文件定义了vertex和edge对应格式，field之间默认分隔符'\t'，数组默认分隔符','，分隔符可配置

    源数据格式描述的图是有向图，如果要表示无向图，可以在边数据文件增加对应的反向边。

    顶点文件和边文件可以有多个，工具转换的时候会并发读取多个文件。

- 顶点文件示例
    - [Vertex](../testdata/vertex_source/vertex.txt)
    ```
    0	1000	3.0	35	3.5
    1	1001	3.0	55	0,0,0	1.1	1001,1002,1003,1004,1005,1006,1007
    ...
    ```
    按照schema文件约定，
    
    第一行依次是vtype, entity, weight, attrs(cid, price)
    
    第二行依次是vtype, entity, weight, attrs(age,edu,addr,test)

    - [Edge](../testdata/edge_source/edge.txt)
    ```
    0	1000	1001	3.5
    1	1001	1000	5.5	3.5	333	1001,1002,1003,1004,1005,1006,1007	100000
    ...
    ```
    按照schema文件约定，
    
    第一行依次是etype, entity_1, entity_2, weight
    
    第二行依次是etype, entity_1, entity_2, weight, attrs(discounts,purchase_num,test,attr1)


## 生成文件说明  
- 二进制文件  
通过图源数据文件，运行转换工具，会转换成galileo能够加载的[二进制文件](../testdata/binary)

- 文件名说明
文件名一般的格式为：edge_0_0.dat或者vertex_0_0.dat,其中第一个“0”代表二进制文件分区的patition序号，第二个“0”，是用来进一步拆分某个patition文件。

