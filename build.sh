#!/bin/bash

source_dir=$(pwd)
build_dir=${source_dir}/build
mkdir -p ${build_dir}
cd ${build_dir}

install_prefix=$(python -c "import sys; print(sys.prefix)")

cmake_args="-DCMAKE_BUILD_TYPE=Release"
cmake_args+=" -DCMAKE_INSTALL_PREFIX=${install_prefix}"
cmake_args+=" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${install_prefix}/lib"
cmake_args+=" -DCMAKE_INSTALL_RPATH=${install_prefix}/lib"
cmake_args+=" -DPYTHON_EXECUTABLE=$(which python)"

cmake ${source_dir} ${cmake_args}
cmake --build . --config Release -- -j$(nproc)
make install
