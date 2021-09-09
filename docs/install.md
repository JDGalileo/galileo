# 源码编译安装Galileo

**我们强烈推荐在docker镜像中源码编译安装Galileo**，可以免去安装依赖包的烦恼。

1. 启动Galileo的[docker开发镜像](https://hub.docker.com/r/jdgalileo/galileo)

CPU版本

```
docker run -it jdgalileo/galileo:devel-cpu bash
```

GPU版本

```
docker run -it jdgalileo/galileo:devel-gpu bash
```

Galileo的docker开发镜像中已经安装了galileo所需的依赖包，包括

- gcc 8.4.0
- python 3.8
- gflags 2.2.2
- glog 0.4.0
- zookeeper 3.5.6
- protobuf 3.9.2
- BRPC 0.9.6
- Tensorflow 2.3.0
- PyTorch 1.6.0

详见[Dockerfile](../docker/devel.Dockerfile)。

2. 克隆galileo的源码仓库
```
git clone https://github.com/JDGalileo/galileo.git
```
3. 编译安装
```
cd galileo
python3 setup.py build
python3 setup.py install
```
4. 验证
```
python3 -c "import galileo; print(galileo.__version__)"
```

[pip或conda安装Galileo](pip.md)

