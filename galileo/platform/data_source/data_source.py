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
import numpy as np
from glob import glob
from galileo.platform.data_source.utils import files_exists, file_exists
from galileo.platform.log import log
from galileo.platform.export import export


@export()
class DataSource:
    def __init__(self, root_dir, name, **kwargs):
        if isinstance(root_dir, str):
            root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.root_dir = root_dir
        self.name = name.lower()
        self.slice_count = kwargs.get('slice_count', 1)
        self.worker_count = kwargs.get('worker_count', 1)
        self.evaluate_vertex_ids = None
        self.test_vertex_ids = None

        # convert
        self._download()
        self._convert_to_readable()
        self._convert_to_binary()

    @property
    def raw_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def convert_to_txt(self):
        raise NotImplementedError

    def convert_to_schema(self):
        raise NotImplementedError

    @property
    def raw_dir(self):
        return os.path.join(self.root_dir, 'raw')

    @property
    def raw_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def output_dir(self):
        return os.path.join(self.root_dir, self.name, 'output')

    @property
    def schema_path(self):
        return os.path.join(self.output_dir, "schema.json")

    @property
    def edge_txt_dir(self):
        return os.path.join(self.output_dir, "edge_source")

    @property
    def edge_txt_path(self):
        return os.path.join(self.edge_txt_dir, "edge.txt")

    @property
    def vertex_txt_dir(self):
        return os.path.join(self.output_dir, "vertex_source")

    @property
    def vertex_txt_path(self):
        return os.path.join(self.vertex_txt_dir, "vertex.txt")

    @property
    def binary_dir(self):
        return os.path.join(self.output_dir, "binary")

    @property
    def evaluate_vertex_ids_path(self):
        return os.path.join(self.output_dir, "evaluate_vertex_ids.npy")

    @property
    def test_vertex_ids_path(self):
        return os.path.join(self.output_dir, "test_vertex_ids.npy")

    def get_evaluate_vertex_ids(self):
        return np.load(self.evaluate_vertex_ids_path, allow_pickle=False)

    def get_test_vertex_ids(self):
        return np.load(self.test_vertex_ids_path, allow_pickle=False)

    def _download(self):
        if files_exists(self.raw_paths):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _convert_to_readable(self):
        if files_exists([
                self.schema_path,
                self.vertex_txt_path,
                self.edge_txt_path,
        ]):
            return
        if self.evaluate_vertex_ids is not None and file_exists(
                self.evaluate_vertex_ids_path):
            return
        if self.test_vertex_ids is not None and file_exists(
                self.test_vertex_ids_path):
            return

        os.makedirs(self.edge_txt_dir, exist_ok=True)
        os.makedirs(self.vertex_txt_dir, exist_ok=True)
        self.convert_to_schema()
        self.convert_to_txt()
        if self.evaluate_vertex_ids is not None:
            np.save(self.evaluate_vertex_ids_path,
                    np.array(self.evaluate_vertex_ids),
                    allow_pickle=False)
        if self.test_vertex_ids is not None:
            np.save(self.test_vertex_ids_path,
                    np.array(self.test_vertex_ids),
                    allow_pickle=False)

    def _convert_to_binary(self):
        files = glob(os.path.join(self.binary_dir, '*.dat'))
        if files and len(files) == self.slice_count * 2:
            return
        os.makedirs(self.binary_dir, exist_ok=True)

        import galileo.framework.pywrap.py_convertor as convertor
        conf = convertor.Config()
        conf.slice_count = self.slice_count
        conf.worker_count = self.worker_count
        conf.schema_path = self.schema_path
        conf.vertex_source_path = self.vertex_txt_dir
        conf.edge_source_path = self.edge_txt_dir
        conf.vertex_binary_path = self.binary_dir
        conf.edge_binary_path = self.binary_dir
        convertor.start_convert(conf)
        log.info(f'convert {self.name} to binary done')
