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


def get_time_str(time):
    if time >= 3600:
        return f'{time/3600:.2f}hour'
    if time >= 60:
        return f'{time/60:.2f}min'
    if time >= 1 or time == 0:
        return f'{time:.1f}s'
    if time >= 1e-3:
        return f'{time*1e3:.0f}ms'
    return f'{time*1e6:.0f}us'
