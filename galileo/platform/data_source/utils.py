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
import tarfile
import zipfile
from urllib.request import urlopen
from galileo.platform.export import export


@export()
def files_exists(files):
    return all([os.path.exists(f) for f in files])


@export()
def file_exists(file):
    return os.path.exists(file)


@export()
def download_url(url, folder):
    '''Downloads the content of an URL to a specific folder'''
    name = os.path.split(url)[1]
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path
    os.makedirs(folder, exist_ok=True)
    print(f'Download from {url}')
    response = urlopen(url)
    CHUNK_SIZE = 4 * 1024
    with open(path, 'wb') as f:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
    return path


@export()
def extract_tar(path, folder, mode='r:gz'):
    with tarfile.open(path, mode) as f:
        f.extractall(folder)
