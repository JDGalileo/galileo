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

source /root/.bashrc

set -e -u

cur_dir=$(cd $(dirname $0);pwd)
project_dir=$(dirname $cur_dir)
libs_dir="$project_dir/galileo/framework/libs"

BUILD_TARGET=${BUILD_TARGET:-cpu}

function copy_deps() {
	name=$(basename $1)
	if [ ! -f "$libs_dir/$name" ];then
		cp $1 $libs_dir
	fi
}

function build_wheel() {
	package_name=$1
	pushd $project_dir
	sed -i "s/package_name = 'jdgalileo'/package_name = '${package_name}'/" setup.py
	python3 setup.py build
	python3 setup.py develop -r

	# copy deps
	copy_deps /usr/local/lib/libgflags.so.2.2.2
	copy_deps /usr/local/lib/libzookeeper_mt.so.2.0.0
	copy_deps /usr/local/lib64/libglog.so.0.4.0
	copy_deps /usr/local/lib64/libleveldb.so.1.22.0
	copy_deps /usr/local/lib64/libprotobuf.so.3.9.2.0
	copy_deps /usr/local/lib64/libprotoc.so.3.9.2.0
	copy_deps /usr/local/lib64/libbrpc.so
	python3 setup.py sdist bdist_wheel
	sed -i "s/package_name = '${package_name}'/package_name = 'jdgalileo'/" setup.py
	popd
}

if [[ $BUILD_TARGET = "gpu" ]];then
	build_wheel jdgalileo-gpu
else
	build_wheel jdgalileo
	build_wheel jdgalileo-cpu
fi
