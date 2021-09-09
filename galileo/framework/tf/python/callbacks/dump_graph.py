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

import tensorflow as tf
from galileo.platform.export import export


@export('galileo.tf')
class DumpGraphCallback(tf.keras.callbacks.Callback):
    '''
    write keras graph to pbtxt in graph mode
    use tf.keras.callbacks.TensorBoard in eager mode instead
    '''
    def __init__(self, summary_dir=None):
        super().__init__()
        self.summary_dir = summary_dir

    def on_train_begin(self, logs=None):
        if not tf.executing_eagerly() and self.summary_dir:
            from tensorflow.python.keras import backend as K
            graph = K.get_session().graph
            tf.io.write_graph(graph, self.summary_dir, 'graph.pbtxt')
