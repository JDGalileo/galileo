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

version=${1:-1.0.0}
echo build galileo ${version}

# base
docker build -t jdgalileo/galileo:base-cpu -f base.Dockerfile \
--build-arg INSTALL_CUDA=0 \
--build-arg BASE_IMAGE=quay.io/pypa/manylinux2014_x86_64:latest .

docker build -t jdgalileo/galileo:base-gpu -f base.Dockerfile \
--build-arg INSTALL_CUDA=1 \
--build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-devel-centos7 .

# devel
docker build -t jdgalileo/galileo:devel-cpu -f devel.Dockerfile \
--build-arg TARGET=cpu .
docker build -t jdgalileo/galileo:devel-gpu -f devel.Dockerfile \
--build-arg TARGET=gpu .

# include galileo package
docker build -t jdgalileo/galileo:${version}-cpu -f galileo.Dockerfile \
--build-arg TARGET=cpu --build-arg VERSION=${version} .
docker build -t jdgalileo/galileo:${version}-gpu -f galileo.Dockerfile \
--build-arg TARGET=gpu --build-arg VERSION=${version} .
