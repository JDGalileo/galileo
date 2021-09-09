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

import os
import argparse
import json
from galileo.platform.export import export
from galileo.platform.default_values import DefaultValues
from galileo.platform.print_version import print_version
from galileo.platform.log import log


@export()
def start_service(root_dir,
                  zk_server=DefaultValues.ZK_SERVER,
                  zk_path=DefaultValues.ZK_PATH,
                  shard_index=0,
                  shard_count=1,
                  hdfs_addr='',
                  hdfs_port=0,
                  thread_num=2,
                  port=0,
                  daemon=False):
    from galileo.framework.pywrap import py_service as service
    conf = service.Config()
    conf.schema_path = os.path.join(root_dir, 'schema.json')
    conf.data_path = os.path.join(root_dir, 'binary')
    conf.hdfs_addr = hdfs_addr
    conf.hdfs_port = hdfs_port
    conf.shard_index = shard_index
    conf.shard_count = shard_count
    conf.thread_num = thread_num
    conf.zk_addr = zk_server
    conf.zk_path = zk_path
    service.start(conf, port, daemon)


@export()
def define_service_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Galileo service')

    parser.add_argument('--data_source_name',
                        '--dataset_name',
                        choices=['ppi', 'cora', 'citeseer', 'pubmed'],
                        type=str,
                        help='public data source name')
    parser.add_argument('--data_source_cache_dir',
                        default='./.data',
                        type=str,
                        help='data source cache dir')
    parser.add_argument('--data_path',
                        '--data_dir',
                        type=str,
                        help='graph data path')
    parser.add_argument('--hdfs_addr',
                        default='',
                        type=str,
                        help='hdfs address')
    parser.add_argument('--hdfs_port', default=0, type=int, help='hdfs port')
    parser.add_argument('--shard_index',
                        default=0,
                        type=int,
                        help='shard index')
    parser.add_argument('--shard_num', default=1, type=int, help='shard num')
    parser.add_argument('--thread_num',
                        default=2,
                        type=int,
                        help='thread number for rpc server')
    parser.add_argument('--zk_server',
                        '--tge_zk_addr',
                        default=DefaultValues.ZK_SERVER,
                        type=str,
                        help='zookeeper address')
    parser.add_argument('--zk_path',
                        '--tge_zk_path',
                        default=DefaultValues.ZK_PATH,
                        type=str,
                        help='zookeeper path')
    parser.add_argument('--role',
                        '--task_type',
                        default='engine_and_worker',
                        type=str,
                        help='task role or type')
    parser.add_argument('--port',
                        '--service_port',
                        default=0,
                        type=int,
                        help='rpc service port')
    parser.add_argument('--daemon', action='store_true', help='service daemon')

    return parser


@export()
def start_service_from_args(args):
    r'''
    start service when role is engine or service
    '''
    print_version()
    if args.role not in ['engine', 'service', 'engine_and_worker']:
        log.info(f'We do not start service for role {args.role}')
        return
    # check for TF_CONFIG, ps don't start service
    if 'TF_CONFIG' in os.environ:
        config = json.loads(os.environ['TF_CONFIG'])
        task = config.get('task')
        if task:
            task_type = task.get('type')
            if task_type == 'ps':
                log.info('We do not start service for ps')
                return
    if args.data_source_name and args.data_path:
        raise RuntimeError(
            'data_source_name and data_path is mutually exclusive')
    if 'engine_and_worker' == args.role:
        log.info(f'Start a daemon service')
        args.daemon = True
    if args.data_source_name:
        # download data source -> convert to binary data
        from galileo.platform.data_source import get_data_source
        ds = get_data_source(output_path=args.data_source_cache_dir,
                             data_source_name=args.data_source_name,
                             slice_count=args.shard_num)
        args.data_path = ds.output_dir
    if args.data_path is None:
        raise RuntimeError('data_source_name or data_path must be specified')
    log.info(f'starting graph service {args.shard_index+1}/{args.shard_num}')
    # start graph service
    start_service(args.data_path,
                  zk_server=args.zk_server,
                  zk_path=args.zk_path,
                  hdfs_addr=args.hdfs_addr,
                  hdfs_port=args.hdfs_port,
                  shard_index=args.shard_index,
                  shard_count=args.shard_num,
                  thread_num=args.thread_num,
                  port=args.port,
                  daemon=args.daemon)
