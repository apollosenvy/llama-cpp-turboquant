#!/usr/bin/env bash
# Usage: build.sh <llama.cpp build dir>   Builds ./fusetest against the shared ggml libraries of that build.
B=$(readlink -f "${1:?usage: build.sh <build dir>}")
cd "$(dirname "$0")"
g++ -O2 -std=c++17 fusetest.cpp -I../../ggml/include -L"$B/bin" -lggml -lggml-hip -lggml-base -Wl,-rpath,"$B/bin" -o fusetest
