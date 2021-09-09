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

from .version import __version__
from .log import log
from .export import export

__logo = r'''
   ______ ___     __     ____ __     ______ ____
  / ____//   |   / /    /  _// /    / ____// __ \
 / / __ / /| |  / /     / / / /    / __/  / / / /
/ /_/ // ___ | / /___ _/ / / /___ / /___ / /_/ /
\____//_/  |_|/_____//___//_____//_____/ \____/
'''


@export()
def print_version():
    log.info(f'\nWelcome to\n{__logo}\nversion {__version__}')
