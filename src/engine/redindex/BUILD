package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "RedIndex",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    hdrs = glob([
        "*.h"
    ]),
    alwayslink = True,
    deps = [
        "//src/core/algorithm:algorithm",
    ],
    linkopts = [
        '-ldl',
    ],
)