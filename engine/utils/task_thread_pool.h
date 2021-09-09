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

#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace galileo {
namespace utils {

typedef std::function<void()> TaskFunc;

class TaskThread {
 public:
  TaskThread(TaskFunc fn) : task_thread_(fn) {}

  ~TaskThread() {
    if (Joinable()) {
      Join();
    }
  }

  bool Joinable() { return task_thread_.joinable(); }

  void Join() {
    if (Joinable()) {
      task_thread_.join();
    }
  }

 private:
  std::thread task_thread_;
};

class TaskQueue {
 public:
  TaskQueue() {}
  TaskFunc Pop() {
    std::unique_lock<std::mutex> lock(mu_);
    auto element = tasks_.front();
    tasks_.pop();
    return element;
  }

  void Push(const TaskFunc& element) {
    std::unique_lock<std::mutex> lock(mu_);
    tasks_.push(element);
  }
  bool IsEmpty() {
    std::unique_lock<std::mutex> lock(mu_);
    return tasks_.empty();
  }
  void Clear() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!tasks_.empty()) tasks_.pop();
  }

 private:
  std::mutex mu_;
  std::queue<TaskFunc> tasks_;
};

class TaskThreadPool {
 public:
  TaskThreadPool() {}

  ~TaskThreadPool() {}

  void AddTask(TaskFunc fn) { task_queues_.Push(fn); }
  void Start(size_t num_threads) {
    size_t max_thread_num = std::thread::hardware_concurrency();
    if (num_threads > max_thread_num) num_threads = max_thread_num;
    for (size_t i = 0; i < num_threads; ++i) {
      task_threads_.emplace_back(new TaskThread([this, i]() { Loop(i); }));
    }
  }

  void Wait() {
    for (auto& task_thread : task_threads_) {
      task_thread->Join();
    }
  }
  void ShutDown() {
    Wait();
    task_threads_.clear();
    task_queues_.Clear();
  }

 private:
  void Loop(size_t i) {
    while (!task_queues_.IsEmpty()) {
      TaskFunc fn = task_queues_.Pop();
      fn();
    }
  }

 private:
  std::vector<std::unique_ptr<TaskThread>> task_threads_;
  TaskQueue task_queues_;
};

}  // namespace utils
}  // namespace galileo
