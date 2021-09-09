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

ARG BASE_IMAGE
ARG INSTALL_CUDA
FROM ${BASE_IMAGE}

COPY base_deps.sh /tmp/
ENV INSTALL_CUDA=${INSTALL_CUDA} MAX_JOBS=16
RUN bash /tmp/base_deps.sh && rm -f /tmp/base_deps.sh
