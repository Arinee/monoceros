cc_library(
    name = "faiss",
    srcs = glob(
        ["*.cpp", "*.h"]
    ),
    hdrs = glob([
        "*.h"
    ]),
    copts = [
        "-I./external/faiss/ -std=c++11 -DFINTEGER=int -fPIC -m64 -Wall -g -O3 -fopenmp -Wno-sign-compare -msse4 -mpopcnt"
    ],
    linkopts = ["-fopenmp -lopenblaso -llapack"],
    include_prefix = "faiss/",
    visibility = ["//visibility:public"],
)
