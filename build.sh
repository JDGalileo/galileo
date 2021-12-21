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

export JAVA_HOME=/usr/lib/jvm/java
export CLASSPATH=$(/opt/hadoop/bin/hadoop classpath --glob)
export PATH=/opt/python/cp38-cp38/bin:/usr/lib/jvm/java/bin:/usr/local/zookeeper/bin:/opt/hadoop/bin:$PATH
export LD_LIBRARY_PATH=/lib64:/usr/local/lib:/usr/local/lib64:/usr/lib/jvm/java/jre/lib/amd64/server:/opt/hadoop/lib/native:$LD_LIBRARY_PATH
export MAX_JOBS=8
ldconfig

python3 setup.py build
python3 setup.py install
