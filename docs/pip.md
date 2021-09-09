# 安装Galileo
我们提供了pip源和conda源安装包，也可以从[源码编译进行安装](install.md)。

## pip安装

CPU版本

```
pip3 install jdgalileo
```

GPU版本

```
pip3 install jdgalileo-gpu
```

## conda安装

CPU版本

```
conda install -c jdgalileo jdgalileo
```

GPU版本

```
conda install -c jdgalileo jdgalileo-gpu
```

## 系统要求

python 3.8

Tensorflow>=2.3.0

PyTorch>=1.6.0

## docker安装

**我们强烈推荐在docker镜像中安装Galileo**，可以免去安装依赖包的烦恼。

1. 在类Unix系统（Ubuntu，CentOS，Mac OS等）或Windows虚拟机上[安装docker](https://docs.docker.com/get-docker/)。
2. 启动Galileo的[docker镜像](https://hub.docker.com/r/jdgalileo/galileo)

```
# CPU
docker run -it jdgalileo/galileo:latest bash
# GPU
docker run -it jdgalileo/galileo:latest-gpu bash
```
以上docker镜像中安装了Galileo最新版本，python 3.8, Tensorflow 2.3.0, PyTorch 1.6.0等，详见[Dockerfile](../docker/galileo.Dockerfile)。

3. 验证
```
python3 -c "import galileo; print(galileo.__version__)"
```

4. 运行例子
参见[Galileo图模型例子](../examples/README.md)
