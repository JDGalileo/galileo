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

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <unistd.h>

#include <brpc/server.h>
#include <butil/logging.h>
#include "discovery/register.h"
#include "service/query_service.h"
#include "service/service.h"

namespace galileo {
namespace service {

Service::Service() {}

Service::~Service() {}

void Service::Start(const GraphConfig& config) { this->Start(0, config); }

void Service::Start(int server_port, const GraphConfig& config) {
  brpc::Server server;

  // init graph rpc
  QueryService graph_query_service;
  if (!graph_query_service.Init(config)) {
    LOG(ERROR) << " Init graph service failed";
    return;
  }

  // add service
  if (server.AddService(&graph_query_service,
                        brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << " Fail to add graph query service";
    return;
  }

  // Start the server
  brpc::ServerOptions options;
  options.num_threads = this->_GetThreadNum(config.thread_num);
  LOG(INFO) << " Brpc bthread_concurrency is " << options.num_threads;

  if (server.Start(server_port, &options) != 0) {
    LOG(ERROR) << " Fail to start brpc: " << server_port;
    return;
  }

  // register shard meta
  butil::EndPoint point = server.listen_address();
  std::string server_ip_port = this->_GetIpPort(point.port);
  galileo::discovery::Register regist(config.zk_addr, config.zk_path);
  regist.AddShard({config.shard_index, server_ip_port},
                  graph_query_service.QueryShardMeta());

  LOG(INFO) << " Brpc service start success";

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();
}

int Service::_GetThreadNum(int thread_num) {
  if (thread_num == 0) {
    thread_num = std::thread::hardware_concurrency() * 2;
  } else {
    thread_num = thread_num > 0 && thread_num < 1000
                     ? thread_num
                     : std::thread::hardware_concurrency() * 2;
  }
  return thread_num;
}

std::string Service::_GetIpPort(int port) const {
  std::string ip_addr = "";
  char hname[128];
  struct hostent* hent;
  gethostname(hname, sizeof(hname));
  hent = gethostbyname(hname);
  if (hent->h_addr_list[0]) {
    ip_addr = inet_ntoa(*(struct in_addr*)(hent->h_addr_list[0]));
  }
  return ip_addr + ":" + std::to_string(port);
}

void StartService(const GraphConfig& config, int port, bool daemon) {
  auto run = [](const GraphConfig& conf, int pt) {
    Service server;
    server.Start(pt, conf);
  };
  if (daemon) {
    std::thread thd(run, config, port);
    thd.detach();
  } else {
    run(config, port);
  }
}

}  // namespace service
}  // namespace galileo
