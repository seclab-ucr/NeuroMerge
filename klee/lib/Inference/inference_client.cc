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
#include "../Json/json.cpp"
#include "inference.grpc.pb.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using org::pytorch::serve::grpc::inference::InferenceAPIsService;
using org::pytorch::serve::grpc::inference::PredictionsRequest;
using org::pytorch::serve::grpc::inference::PredictionResponse;

class InferenceClient {
 public:
  InferenceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceAPIsService::NewStub(channel)) {
    // routeguide::ParseDb(db, &feature_list_);
  }

  // void GetResults() {
  //   // Point point;
  //   // Feature feature;
  //   Jsonfile json1;
  //   Results result;
  //   json1.set_contents("hello,world");
  //   GetOneResult(json1, &result);
  //   json1.set_contents("yeah");
  //   GetOneResult(json1, &result);
  // }

  bool infer(nlohmann::json json) {
    ClientContext context;
    PredictionsRequest request;
    request.set_model_name("mergegraph0409");
    auto map = request.mutable_input();
    
    (*map)["body"] = json.dump();
    PredictionResponse* result;
    std::cout << "before Predictions" << std::endl;
    Status status = stub_->Predictions(&context, request, result);
    std::cout << "after Predictions" << std::endl;
    if (!status.ok()) {
      std::cout << "GetFeature rpc failed." << std::endl;
      return false;
    }
    std::cout << "GetFeature rpc succeeded." << std::endl;
    if (result->prediction().empty()) {
      std::cout << "No return "<< std::endl;
    } else {
      std::cout << "Merge: " << result->prediction() << std::endl;
    }
    return true;
  }

  const float kCoordFactor_ = 10000000.0;
  std::unique_ptr<InferenceAPIsService::Stub> stub_;
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
