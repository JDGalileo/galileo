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

# install galileo required packages

MAX_JOBS=${MAX_JOBS:-32}
ZK_URL=${ZK_URL:-https://archive.apache.org/dist/zookeeper/zookeeper-3.5.6/apache-zookeeper-3.5.6.tar.gz}
PYPI_URL=https://mirrors.ustc.edu.cn/pypi/web/simple
GTEST_URL=https://github.com/google/googletest/archive/v1.10.x.tar.gz
GFLAGS_URL=HTTPS://github.com/gflags/gflags/archive/v2.2.2.tar.gz
GLOG_URL=https://github.com/google/glog/archive/v0.4.0.tar.gz
RAPIDJSON_URL=https://github.com/Tencent/rapidjson/archive/master.zip
PYBIND11_URL=https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz
PB_URL=https://github.com/protocolbuffers/protobuf/archive/v3.9.2.tar.gz
BRPC_URL=https://github.com/apache/incubator-brpc/archive/0.9.6.tar.gz
LEVELDB_URL=https://github.com/google/leveldb/archive/1.22.zip

function install_zk_devel() {
	yum -y install ant libtool
	wget -O zookeeper.tar.gz ${ZK_URL}
	tar xf zookeeper.tar.gz
	rm -f zookeeper.tar.gz
	pushd apache-zookeeper-3.5.6
	ant compile_jute
	pushd zookeeper-client/zookeeper-client-c
	export PATH=/usr/bin:$PATH
	autoreconf -if
	./configure --disable-static --without-cppunit --prefix=/usr/local
	make -j${MAX_JOBS} && make install
	popd && popd
	rm -fr /root/.ant
	yum -y remove ant gcc
	yum clean all && rm -rf /var/cache/yum/*
}


function install_all() {
	pip3 install --no-cache-dir cmake
	ln -sf /opt/_internal/cpython-3.8.12/bin/cmake /usr/local/bin/cmake

	wget -O rapidjson.zip ${RAPIDJSON_URL}
	unzip -q rapidjson.zip && rm -f rapidjson.zip
	pushd rapidjson* && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .
	make -j${MAX_JOBS} && make install && popd && rm -fr rapidjson*

    wget -O pybind11.tar.gz ${PYBIND11_URL}
    tar xf pybind11.tar.gz && rm -f pybind11.tar.gz
	pushd pybind11*
	cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTING=OFF .
	make -j${MAX_JOBS} && make install && popd && rm -fr pybind11*

    wget -O gtest.tar.gz ${GTEST_URL}
    tar xf gtest.tar.gz && rm -f gtest.tar.gz
	pushd googletest*
	cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local \
	-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .
	make -j${MAX_JOBS} && make install && popd && rm -fr googletest*

    wget -O gflags.tar.gz ${GFLAGS_URL}
    tar xf gflags.tar.gz && rm -f gflags.tar.gz
	pushd gflags*
	cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .
	make -j${MAX_JOBS} && make install && popd && rm -fr gflags*

    wget -O glog.tar.gz ${GLOG_URL} && tar xf glog.tar.gz && rm -f glog.tar.gz
	pushd glog*
	cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .
	make -j${MAX_JOBS} && make install && popd && rm -fr glog*

	wget -O pb.tar.gz ${PB_URL} && tar xf pb.tar.gz && rm -f pb.tar.gz
	pushd protobuf* && cmake -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON \
	-Dprotobuf_BUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_PROTOC_BINARIES=ON \
	-DCMAKE_INSTALL_PREFIX=/usr/local \
	-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" ./cmake
	make -j${MAX_JOBS} && make install && popd && rm -fr protobuf*

    wget -O 1.22.zip ${LEVELDB_URL} && unzip -q 1.22.zip
	rm -f 1.22.zip && pushd leveldb-1.22
	cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
	-DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .
	make -j${MAX_JOBS} && make install && popd && rm -fr leveldb*

    wget -O brpc.tar.gz ${BRPC_URL} && tar xf brpc.tar.gz && rm -f brpc.tar.gz
	pushd incubator*
	sed -i "s/set(CMAKE_CPP_FLAGS \"\${DEFINE_CLOCK_GETTIME}/set(CMAKE_CPP_FLAGS \"-D_GLIBCXX_USE_CXX11_ABI=0 \${DEFINE_CLOCK_GETTIME}/" \
	CMakeLists.txt tools/CMakeLists.txt
	cmake -DWITH_GLOG=ON -DWITH_DEBUG_SYMBOLS=OFF -DDOWNLOAD_GTEST=OFF \
	-DCMAKE_INSTALL_PREFIX=/usr/local .
	make -j${MAX_JOBS} && make install && popd && rm -fr incubator*
}

function install_py() {
    pip3 config set global.index-url ${PYPI_URL}
    pip3 install pytest pytest-cov pytest-runner pytest-remotedata
    pip3 cache purge
}

install_zk_devel
install_all
install_py
