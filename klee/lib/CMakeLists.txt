#===------------------------------------------------------------------------===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

#pkg-config --cflags --libs protobuf grpc++ grpc

 

  # # Find gRPC installation
  # # Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
  # find_package(gRPC CONFIG REQUIRED)
  # message(STATUS "Using gRPC ${gRPC_VERSION}")

  # set(_GRPC_GRPCPP gRPC::grpc++)
  # if(CMAKE_CROSSCOMPILING)
  #   find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
  # else()
  #   set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  # endif()

add_subdirectory(ToolLib)
add_subdirectory(Json)
add_subdirectory(Basic)
add_subdirectory(Support)
add_subdirectory(Expr)
add_subdirectory(Solver)
add_subdirectory(Module)
add_subdirectory(Core)
# add_subdirectory(Inference)
