#!/bin/bash

set -xe 

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"


compile() {
    echo "run build"
    bazel build --config=release //src/engine/redindex:RedIndex
}

ut() {
    echo "run ut"
    bazel test --config=release --test_timeout=600 //tests/engine/redindex:all_test
    bazel test --config=release --test_timeout=900 //tests/core:all_test
}

case $1 in 
    compile )
        compile
        ;;
    ut )
        ut
        ;;
esac
