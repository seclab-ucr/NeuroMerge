protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`  inference.proto 
protoc --cpp_out=. inference.proto 

protoc -I serve/frontend/server/src/main/resources/proto/ --grpc_out=serve/frontend/server/src/main/resources/proto  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` serve/frontend/server/src/main/resources/proto/inference.proto
protoc -I serve/frontend/server/src/main/resources/proto/ --cpp_out=serve/frontend/server/src/main/resources/proto/ serve/frontend/server/src/main/resources/proto/inference.proto