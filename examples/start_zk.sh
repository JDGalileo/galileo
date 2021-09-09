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

# check and start zookeeper
zkc=$(which zkCli.sh)
[ $? -ne 0 ] && exit
zks=$(which zkServer.sh)
[ $? -ne 0 ] && exit
${zkc} -server 127.0.0.1:2181 ls / > /dev/null 2>&1
if [ $? -ne 0 ];then
    ${zks} stop > /dev/null 2>&1
    echo "start zookeeper"
    ${zks} start
fi
