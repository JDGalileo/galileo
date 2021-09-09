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

from .backend import is_tf, is_pytorch, set_backend


class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        raise RuntimeError(f'{name} is deprecated, use galileo.unify.api instead')
        cls = super().__new__(cls, name, bases, attrs)
        cls._cached_models = {}
        return cls

    def select_base_model(cls, backend):
        if backend not in cls._cached_models:
            set_backend(backend)
            if is_tf():
                import galileo.framework.tf.python as models
            elif is_pytorch():
                import galileo.framework.pytorch.python as models
            else:
                raise RuntimeError('Only support backend tf or pytorch')
            if hasattr(models, cls.__name__):
                base_name = cls.__name__
            elif hasattr(models, cls.__base__.__name__):
                base_name = cls.__base__.__name__
            else:
                raise RuntimeError(f'No base class found for {cls.__name__}')
            base = getattr(models, base_name)

            def init(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)
                cls.__init__(self, *args, **kwargs)

            attrs = cls.__dict__.copy()
            attrs['__init__'] = init
            base_model = type(cls.__name__, (base, ), attrs)
            cls._cached_models[backend] = base_model
        return cls._cached_models[backend]


class baseModel(object):
    def __new__(cls, *args, **kwargs):
        backend = kwargs.get('backend')
        assert backend, 'must set backend'
        new_cls = cls.select_base_model(backend)
        ins = new_cls(*args, **kwargs)
        return ins


class Supervised(baseModel, metaclass=ModelMeta):
    r'''
    supervised network embedding model

    user model should inherit this class

    Methods that the subclass must implement:
        encoder

    example:
        class model(galileo.nn.Supervised):
            def encoder(self, inputs):
                pass

    args:
        label_dim: label dim
        embedding_dim: embedding dim for dense layer
        num_classes: num of class
        metric: metric name, default is f1_score
    '''


class Unsupervised(baseModel, metaclass=ModelMeta):
    r'''
    unsupervised network embedding model

    user model should inherit this class

    Methods that the subclass must implement:
        target_encoder
        context_encoder

    example:
        class model(galileo.nn.Unsupervised):
            def target_encoder(self, inputs):
                pass
            def context_encoder(self, inputs):
                pass
    '''


class Embedding(baseModel, metaclass=ModelMeta):
    r'''
    wrap Embedding in tensorflow and pytorch

    args:
        embedding_size: size of the dictionary of embeddings
        embedding_dim: the size of each embedding vector
    '''
    def __init__(self, *args, **kwargs):
        baseModel.__init__(self)


class Dense(baseModel, metaclass=ModelMeta):
    r'''
    wrap dense layer in tensorflow and pytorch

    args:
        input_dim: size of each input sample
        feature_dim: size of each output sample
        bias: whether the layer uses a bias vector
    '''
    def __init__(self, *args, **kwargs):
        baseModel.__init__(self)
