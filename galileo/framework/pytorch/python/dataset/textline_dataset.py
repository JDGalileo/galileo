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
import torch
from torch.utils.data import Dataset
from galileo.platform.export import export


@export('galileo.pytorch')
class TextLineDataset(Dataset):
    '''
    Only support local files, NOT hdfs files

    args
        input_file: is a full filename
        input_file_num_cols: num cols of csv files
    '''
    def __init__(self, input_file, input_file_num_cols, **kwargs):
        super().__init__()

        assert os.path.exists(input_file), \
            'the inputfile : {} is not existed'.format(input_file)
        self.all_lines = [lines.rstrip('\n') for lines in open(input_file)]

        self.input_file_num_cols = input_file_num_cols
        assert input_file_num_cols > 0, 'input_file_num_cols should be > 0'

    def get(self, idx):
        line = self.all_lines[idx]
        return self.parse_line(line)

    def parse_line(self, line):
        line_data = line.split(',')
        assert self.input_file_num_cols == len(line_data), \
            f'Error line: {line}'
        line_data = torch.LongTensor(list(map(int, line_data)))
        line_data = torch.reshape(line_data, [-1, 1])
        return line_data

    def __len__(self):
        return len(self.all_lines)

    def __getitem__(self, idx):
        return self.get(idx)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))
