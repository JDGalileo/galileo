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
from galileo.platform.log import log
from galileo.framework.python.convert import convert
from galileo.platform.print_version import print_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertex_source_path',
                        required=True,
                        type=str,
                        help='vertex source path')
    parser.add_argument('--edge_source_path',
                        required=True,
                        type=str,
                        help='edge source path')
    parser.add_argument('--schema_path',
                        required=True,
                        type=str,
                        help='schema path, json file')
    parser.add_argument('--output_binary_path',
                        required=True,
                        type=str,
                        help='binary output path')
    parser.add_argument('--partition_num',
                        default=1,
                        type=int,
                        help='partition number')
    parser.add_argument('--parallel',
                        default=1,
                        type=int,
                        help='work thread parallel number')
    parser.add_argument('--hdfs_addr',
                        default='',
                        type=str,
                        help='hdfs address')
    parser.add_argument('--hdfs_port', default=0, type=int, help='hdfs port')
    parser.add_argument('--field_separator',
                        default='\t',
                        type=str,
                        help='field separator, only one char')
    parser.add_argument('--array_separator',
                        default=',',
                        type=str,
                        help='array separator, only one char')
    args, _ = parser.parse_known_args()
    print_version()
    log.info(f'Galileo converter args {vars(args)}')
    convert(**vars(args))
    out_path = os.path.dirname(args.output_binary_path)
    log.info(f'Galileo converter done output path {args.output_binary_path}\n'
             f'data_path for galileo service is {out_path}')


if __name__ == '__main__':
    main()
