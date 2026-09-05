### LLVM 6.0
```shell
cd build
wget https://releases.llvm.org/6.0.0/llvm-6.0.0.src.tar.xz
wget https://releases.llvm.org/6.0.0/cfe-6.0.0.src.tar.xz
tar xf ./llvm-6.0.0.src.tar.xz
tar xf ./cfe-6.0.0.src.tar.xz
mv cfe-6.0.0.src ./llvm-6.0.0.src/tools/clang
cd ../build
mkdir llvm-build && cd llvm-build
cmake -G "Unix Makefiles" \
   -DLLVM_ENABLE_ASSERTIONS=true \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DLLVM_ENABLE_RTTI=ON \
   -DCMAKE_INSTALL_PREFIX:PATH=../../install \
   ../../build/llvm-6.0.0.src
make -j12
make install
cd ../..
```

### z3
```shell
cd build
git clone https://github.com/Z3Prover/z3.git
cd z3
mkdir ../../build/z3-build && cd ../../build/z3-build
cmake -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../../install \
  ../../build/z3/
make -j12
make install
cd ../..
```

###  Build uClibc and the POSIX environment model

```shell
cd build
git clone https://github.com/klee/klee-uclibc.git
cd klee-uclibc
./configure --make-llvm-lib
make -j2
cd ../..
```

### klee

```shell
sudo apt-get install libsqlite3-dev
sudo apt install doxygen
cd source
cd klee-based
mkdir ../../build/klee-dataset-pipeline-build && cd ../../build/klee-dataset-pipeline-build
cmake -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_CONFIG_BINARY=../../install/bin/llvm-config \
  -DENABLE_SOLVER_Z3=ON \
  -DZ3_INCLUDE_DIRS=../../install/include \
  -DZ3_LIBRARIES=../../install/lib/libz3.so \
  -DENABLE_UNIT_TESTS=OFF \
  -DENABLE_SYSTEM_TESTS=OFF \
  -DENABLE_TCMALLOC=OFF \
  -DENABLE_POSIX_RUNTIME=ON \
  -DENABLE_KLEE_UCLIBC=ON  \
  -DKLEE_UCLIBC_PATH=../../build/klee-uclibc \
  -G "CodeBlocks - Unix Makefiles" \
  ../../source/klee-dataset-pipeline
make -j64
cd ../..
```

### build & test coreutils

https://klee.github.io/tutorials/testing-coreutils/