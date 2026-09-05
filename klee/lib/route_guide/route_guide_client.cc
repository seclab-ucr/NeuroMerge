/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>


#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#ifdef BAZEL_BUILD
#include "examples/protos/route_guide.grpc.pb.h"
#else
#include "route_guide.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using routeguide::RouteGuide;
using routeguide::Jsonfile;
using routeguide::Results;

class RouteGuideClient {
 public:
  RouteGuideClient(std::shared_ptr<Channel> channel)
      : stub_(RouteGuide::NewStub(channel)) {
    // routeguide::ParseDb(db, &feature_list_);
  }

  void GetResults() {
    // Point point;
    // Feature feature;
    Jsonfile json1;
    Results result;
    json1.set_contents("hello,world");
    GetOneResult(json1, &result);
    json1.set_contents("yeah");
    GetOneResult(json1, &result);
  }

  bool GetOneResult(const Jsonfile& jsonfile, Results* result) {
    ClientContext context;
    Status status = stub_->infer(&context, jsonfile, result);
    if (!status.ok()) {
      std::cout << "GetFeature rpc failed." << std::endl;
      return false;
    }

    if (result->res().empty()) {
      std::cout << "No return "<< std::endl;
    } else {
      std::cout << "Merge: " << result->res() << std::endl;
    }
    return true;
  }

  const float kCoordFactor_ = 10000000.0;
  std::unique_ptr<RouteGuide::Stub> stub_;
};

// int main(int argc, char** argv) {
//   // Expect only arg: --db_path=path/to/route_guide_db.json.
// std::cout << "start" << std::endl;
//   std::string db = routeguide::GetDbFileContent(argc, argv);
//   RouteGuideClient guide(
//       grpc::CreateChannel("localhost:50051",
//                           grpc::InsecureChannelCredentials()));

//   std::cout << "-------------- GetFeature --------------" << std::endl;
//   guide.GetResults();
//   return 0;
// }
