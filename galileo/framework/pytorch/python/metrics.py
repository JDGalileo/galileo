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

import torch
from galileo.platform.export import export


def mrr(logits, negative_logits):
    r'''
    Mean reciprocal rank score.
    '''
    scores = torch.cat((negative_logits, logits), dim=-1)
    k = scores.shape[-1]
    ranks_idx = scores.topk(k)[1]
    ranks = (-ranks_idx).topk(k)[1]
    ranks = ranks.view(-1, k)
    return (ranks[:, -1] + 1).float().reciprocal().mean()


def acc(y_true, y_pred):
    r'''
    accuracy
    '''
    t = y_true.max(dim=1)[1]
    p = y_pred.max(dim=1)[1]
    return (t == p).sum().item() / len(y_true)


def f1_score(y_true, y_pred, epsilon=1e-7):
    r'''
    Calculate micro F1 score.

    NOTE: use this to evaluate for the whole epoch, not for one batch

    args:
        y_true: torch.Tensor
        y_pred: torch.Tensor
    return (f1, precision, recall) tuple of torch.Tensor
        (dim == 1 and 0 <= val <= 1)
    '''
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1, precision, recall


def cosine(y_true, y_pred):
    '''
    return tensor with none reduction
    '''
    cos = torch.nn.CosineSimilarity(dim=-1)
    return cos(y_true, y_pred)


metrics_dict = {k: v for k, v in globals().items() if callable(v)}


@export('galileo.pytorch')
def get_metric(name):
    assert name in metrics_dict, f'not support metric {name}'
    return metrics_dict[name]
