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

import math


class Statistics(object):
    def __init__(self, fmt=':f'):
        super().__init__()
        self.fmt = fmt
        self.reset()

    def reset(self):
        # statistics
        self.val = 0
        self.avg = 0
        self.sum_ = 0
        self.count = 0
        self.min_ = math.inf
        self.max_ = -math.inf
        self.last_sum = 0
        self.last_count = 0

    def reset_last(self):
        self.last_sum = 0
        self.last_count = 0

    def update(self, val):
        self.val = val
        self.sum_ += val
        self.count += 1
        self.min_ = min(self.min_, val)
        self.max_ = max(self.max_, val)
        self.last_sum += val
        self.last_count += 1

    def get_last_result(self):
        if self.last_count == 0:
            return "no data"
        last_avg = self.last_sum / self.last_count
        self.reset_last()
        fmtstr = '{' + self.fmt + '}'
        return fmtstr.format(last_avg)

    def get_result(self):
        if self.count == 0:
            return "no data"
        avg = self.sum_ / self.count
        return self._get_format_str().format(self.min_, avg, self.max_)

    def _get_format_str(self):
        fmtstr = 'min/mean/max: {' + self.fmt + '}/{' + \
                self.fmt + '}/{' + self.fmt + '}'
        return fmtstr
