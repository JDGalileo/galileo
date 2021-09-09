// Copyright 2020 JD.com, Inc. Galileo Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include <cfloat>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/types.h"
#include "glog/logging.h"

namespace galileo {
namespace common {

namespace {
template <typename T = WeightType>
T UniformRandom() {
  static_assert(std::is_floating_point<T>::value,
                "Only support float point type");
  static thread_local std::default_random_engine e(time(0));
  static thread_local std::uniform_real_distribution<T> u(0., 1.);
  return u(e);
}
}  // namespace

// https://en.wikipedia.org/wiki/Alias_method
template <typename T = WeightType>
class AliasSampler {
 public:
  static_assert(std::is_floating_point<T>::value,
                "Only support float point type");
  using AliasType = uint32_t;
  bool Init(std::vector<T>& distribution);
  inline bool Init(const std::vector<T>& distribution);
  AliasType Sample() const;
  inline size_t Size() const;

 private:
  void _Normalization(std::vector<T>& distribution) const;

 private:
  std::vector<T> prob_;
  std::vector<AliasType> alias_;
};

template <typename T>
bool AliasSampler<T>::Init(std::vector<T>& distribution) {
  if (distribution.size() >= 0xffffffff) {
    return false;
  }
  // normalization sum of distribution to 1
  _Normalization(distribution);

  prob_.resize(distribution.size());
  alias_.resize(distribution.size());
  std::vector<AliasType> smaller, larger;
  smaller.reserve(distribution.size());
  larger.reserve(distribution.size());

  for (size_t i = 0; i < distribution.size(); ++i) {
    prob_[i] = distribution[i] * distribution.size();
    if (prob_[i] < 1.0) {
      smaller.push_back(i);
    } else {
      larger.push_back(i);
    }
  }
  // Construct the probability and alias tables
  AliasType small, large;
  while (!smaller.empty() && !larger.empty()) {
    small = smaller.back();
    smaller.pop_back();
    large = larger.back();
    larger.pop_back();
    alias_[small] = large;
    prob_[large] = prob_[large] + prob_[small] - 1.0;
    if (prob_[large] < 1.0) {
      smaller.push_back(large);
    } else {
      larger.push_back(large);
    }
  }
  while (!smaller.empty()) {
    small = smaller.back();
    smaller.pop_back();
    prob_[small] = 1.0;
  }
  while (!larger.empty()) {
    large = larger.back();
    larger.pop_back();
    prob_[large] = 1.0;
  }
  return true;
}

template <typename T>
bool AliasSampler<T>::Init(const std::vector<T>& distribution) {
  std::vector<T> dist = distribution;
  return Init(dist);
}

template <typename T>
typename AliasSampler<T>::AliasType AliasSampler<T>::Sample() const {
  AliasType roll = floor(prob_.size() * UniformRandom());
  bool coin = UniformRandom() < prob_[roll];
  return coin ? roll : alias_[roll];
}

template <typename T>
size_t AliasSampler<T>::Size() const {
  return prob_.size();
}

template <typename T>
void AliasSampler<T>::_Normalization(std::vector<T>& distribution) const {
  static uint64_t wlog_cur=0;
  double norm_sum = 0.;
  for (auto& dist : distribution) {
    norm_sum += dist;
  }
  if (norm_sum <= DBL_EPSILON and !distribution.empty()) {
    for (size_t i = 0; i < distribution.size(); ++i) {
      distribution[i] = 1. / distribution.size();
    }
    if ((wlog_cur & (wlog_cur-1) == 0)) {
      LOG(WARNING) << " Sum of distribution in AliasSampler is ZERO, "
                   << "but the distribution is not empty, set to 1.0";
    }
    ++wlog_cur;
    return;
  }
  for (size_t i = 0; i < distribution.size(); ++i) {
    distribution[i] /= norm_sum;
  }
}

template <typename Type>
class SimpleSampler {
 public:
  bool Init(const std::vector<Type>& entitys,
            const std::vector<float>& weights);
  Type Sample() const;

  size_t GetSize() const { return weights_.size(); }
  float GetWeight(size_t index) const { return weights_[index]; }

  bool IsEmpty() { return weights_.size() <= 0; }

 private:
  std::vector<Type> entitys_;
  std::vector<WeightType> weights_;
  AliasSampler<WeightType> alias_sampler_;
};

template <typename Type>
bool SimpleSampler<Type>::Init(const std::vector<Type>& entitys,
                               const std::vector<float>& weights) {
  if (entitys.size() != weights.size() || 0 == entitys.size()) {
    LOG(ERROR) << " Param is invalid.entity num:" << entitys.size()
               << " ,weight num:" << weights.size();
    return false;
  }
  entitys_ = entitys;
  weights_ = weights;

  if (!alias_sampler_.Init(weights)) {
    LOG(ERROR) << " Alias_sampler_ Initilize fail.";
    return false;
  }
  return true;
}

template <typename Type>
Type SimpleSampler<Type>::Sample() const {
  auto column = alias_sampler_.Sample();
  return entitys_[column];
}

template <typename ENTITY, typename ENTITYID>
class WeightedSampler {
 public:
  bool Init(std::vector<ENTITY*>& entitys);

  bool Sample(std::pair<ENTITYID, float>& pair) const;

  bool IsEmpty() { return entitys_.size() <= 0; }

 private:
  std::vector<ENTITY*> entitys_;
  AliasSampler<WeightType> alias_sampler_;
};

template <typename ENTITY, typename ENTITYID>
bool WeightedSampler<ENTITY, ENTITYID>::Init(std::vector<ENTITY*>& entitys) {
  entitys_ = std::move(entitys);
  if (entitys_.size() <= 0) {
    return true;
  }
  std::vector<WeightType> weights;
  weights.resize(entitys_.size());
  for (size_t i = 0; i < entitys_.size(); ++i) {
    weights[i] = entitys_[i]->GetWeight();
  }
  if (!alias_sampler_.Init(weights)) {
    LOG(ERROR) << " Alias_sampler_ Initilize fail.";
    return false;
  }
  return true;
}

template <typename ENTITY, typename ENTITYID>
bool WeightedSampler<ENTITY, ENTITYID>::Sample(
    std::pair<ENTITYID, float>& pair) const {
  if (entitys_.size() <= 0) {
    return false;
  }
  auto column = alias_sampler_.Sample();
  pair =
      std::make_pair(entitys_[column]->GetId(), entitys_[column]->GetWeight());
  return true;
}

}  // namespace common
}  // namespace galileo
