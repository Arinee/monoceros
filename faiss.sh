#!/bin/bash

ExitFunc(){
    if [ $1 -ne 0 ]
    then
        echo "FAIL..."
        exit $1
    fi
}
./bin/local_builder -build_input ../tests/faiss_data/input.dat -input_first_sep ';' -dimension 256 -builder_class FaissIndexBuilder -storage_class FileStorage -index_prefix sample.index lib/libfaiss_index.so
ExitFunc $?

./bin/recall -storage_class FileStorage -service_class FaissService -query ../tests/faiss_data/query.dat -input_first_sep ';' -index sample.index lib/libfaiss_index.so
ExitFunc $?

echo "PASS..."
