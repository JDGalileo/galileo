#!/bin/bash
# Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e -u

INSTALL_CUDA=${INSTALL_CUDA:-0}
MAX_JOBS=${MAX_JOBS:-32}
PYPI_URL=https://mirrors.ustc.edu.cn/pypi/web/simple
TORCH_URL=https://download.pytorch.org/whl/torch_stable.html
HADOOP_URL=${HADOOP_URL:-https://downloads.apache.org/hadoop/common/hadoop-3.3.0/hadoop-3.3.0.tar.gz}
ZK_BIN_URL=${ZK_BIN_URL:-https://archive.apache.org/dist/zookeeper/zookeeper-3.5.6/apache-zookeeper-3.5.6-bin.tar.gz}
OPENSSL_URL=${OPENSSL_URL:-http://www.openssl.org/source/openssl-1.1.0c.tar.gz}


function install_gcc() {
    #for nvidia/cuda:10.1-cudnn7-devel-centos7
    GCC_VERSION=8.4.0
    GCC_URL=https://mirrors.ustc.edu.cn/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz
    MAKE_URL=https://mirrors.ustc.edu.cn/gnu/make/make-4.3.tar.gz
    CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.sh

    sed -e 's|^mirrorlist=|#mirrorlist=|g' \
    -e 's|^#baseurl=http://mirror.centos.org/centos|baseurl=https://mirrors.ustc.edu.cn/centos|g' \
    -i.bak /etc/yum.repos.d/CentOS-Base.repo
    yum -y install wget which vim
    yum -y groupinstall 'Development Tools'
    wget -qO make.tar.gz ${MAKE_URL} && tar xf make.tar.gz && rm -f make.tar.gz
    pushd make-4.3 && ./configure --prefix=/usr/local/ && make -j${MAX_JOBS}
    make install && popd && rm -rf make-4.3 && yum -y remove make
    ln -srf /usr/local/bin/make /usr/bin/gmake
    ln -srf /usr/local/bin/make /usr/bin/make

    wget -qO cmake.sh ${CMAKE_URL} && bash cmake.sh --skip-license --prefix=/usr/local && rm -f cmake.sh
    wget -q ${GCC_URL} && tar xf gcc-${GCC_VERSION}.tar.xz && rm -f gcc-${GCC_VERSION}.tar.xz
    pushd gcc-${GCC_VERSION} && ./contrib/download_prerequisites
    ./configure --enable-checking=release --enable-languages=c,c++,obj-c++ --disable-multilib
    make -j${MAX_JOBS} && make install && popd && rm -rf gcc-${GCC_VERSION} && yum -y remove gcc
}

function install_ssl() {
    yum -y install wget zlib zlib-devel
    wget -O openssl-1.1.0c.tar.gz ${OPENSSL_URL}
    tar xf openssl-1.1.0c.tar.gz
    pushd openssl-1.1.0c
    ./config shared zlib
    make -j${MAX_JOBS}
    make install_sw
    popd
    rm -fr openssl-1.1.0*
    yum clean all && rm -rf /var/cache/yum/*
}

function install_zk() {
    yum -y install java-1.8.0-openjdk java-1.8.0-openjdk-devel
    yum clean all && rm -rf /var/cache/yum/*
    echo "export JAVA_HOME=/usr/lib/jvm/java" >> /root/.bashrc
    echo "export CLASSPATH=\$(/opt/hadoop/bin/hadoop classpath --glob)" \
    >> /root/.bashrc

    wget -O hadoop.tar.gz ${HADOOP_URL}
    tar -xf hadoop.tar.gz -C /opt
    rm -f hadoop.tar.gz
    mv /opt/hadoop* /opt/hadoop

    wget -O zookeeper.tar.gz ${ZK_BIN_URL}
    tar xf zookeeper.tar.gz -C /usr/local/
    rm -f zookeeper.tar.gz && mv /usr/local/apache-zookeeper-3.5.6-bin \
    /usr/local/zookeeper
    mkdir -p /usr/local/zookeeper/data
    echo -e "dataDir=/usr/local/zookeeper/data\nclientPort=2181" \
    > /usr/local/zookeeper/conf/zoo.cfg
    echo "JAVA_HOME=/usr/lib/jvm/java" \
    > /usr/local/zookeeper/conf/zookeeper-env.sh
}

function setup_env() {
    paths=""
    libs="/lib64:/usr/local/lib:/usr/local/lib64"
    libs+=":/usr/lib/jvm/java/jre/lib/amd64/server"
    libs+=":/opt/hadoop/lib/native"
    if [ ${INSTALL_CUDA} -ne 0 ];then
        paths+="/usr/local/bin"
        paths+=":/usr/local/anaconda3/bin"
        paths+=":/usr/local/cuda/bin"
        libs+=":/usr/local/anaconda3/lib"
        libs+=":/usr/local/cuda/lib64"
        libs+=":/usr/local/nvidia/lib"
        libs+=":/usr/local/nvidia/lib64"
    fi
    paths+=":/usr/lib/jvm/java/bin"
    paths+=":/usr/local/zookeeper/bin"
    paths+=":/opt/hadoop/bin"
    echo "export PATH=${paths}:\$PATH" >> /root/.bashrc
    echo "export LD_LIBRARY_PATH=${libs}:\$LD_LIBRARY_PATH" >> /root/.bashrc
    echo "export LIBRARY_PATH=${libs}:\$LIBRARY_PATH" >> /root/.bashrc
    echo "export MAX_JOBS=${MAX_JOBS}" >> /root/.bashrc
    set +u
    source /root/.bashrc || true
}

function install_py3() {
    #for nvidia/cuda:10.1-cudnn7-devel-centos7
    ANACONDA_URL=https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    wget -qO anaconda3.sh ${ANACONDA_URL} && bash anaconda3.sh -b -p /usr/local/anaconda3
    rm -f anaconda3.sh && cp /usr/local/anaconda3/lib/libstdc++.so.6.0.26 /lib64
    ln -srf /lib64/libstdc++.so.6.0.26 /lib64/libstdc++.so.6
    ln -srf /lib64/libstdc++.so.6.0.26 /usr/lib64/libstdc++.so.6
}

function install_deps_gpu() {
    #for nvidia/cuda:10.1-cudnn7-devel-centos7
    pip=/usr/local/anaconda3/bin/pip3
    conda=/usr/local/anaconda3/bin/conda
    ${pip} install -i ${PYPI_URL} pip -U
    ${pip} config set global.index-url ${PYPI_URL}
    ${pip} install tensorflow==2.3.0 networkx==2.3 attrs
    ${pip} install torch==1.6.0+cu101 torchvision==0.7.0+cu101 torch-scatter -f ${TORCH_URL}
    ${conda} install -y numpy scipy pyyaml ipython mkl mkl-include scikit-learn
    ${conda} install -c conda-forge -y kazoo py3nvml
    ${conda} clean -ya && ${pip} cache purge
}

function install_deps_cpu() {
    echo "install for python $1"
    pip=/opt/python/$1/bin/pip
    ${pip} config set global.index-url ${PYPI_URL}
    ${pip} install pip -U
    ${pip} install torch==1.6.0+cpu torchvision==0.7.0+cpu -f ${TORCH_URL}
    ${pip} install torch-scatter tensorflow==2.3.0 networkx==2.3 kazoo attrs
    ${pip} cache purge
    ln -sf /opt/python/$1/bin/pip3 /usr/local/bin/pip3
    ln -sf /usr/local/bin/python3.8 /usr/local/bin/python3
}

if [ ${INSTALL_CUDA} -eq 0 ];then
    echo "install cpu version"
    install_deps_cpu cp38-cp38
else
    echo "install gpu version"
    install_gcc
    install_deps_gpu
fi

install_ssl
install_zk
setup_env
