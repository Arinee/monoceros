package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "local_runner",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    copts = [
        "-fopenmp"
    ],
    linkopts = [
        '-ldl',
        "-fopenmp"
    ],
    deps = [
        "//src/core/algorithm:algorithm",
    ],
    data = glob([
        "log.conf",
        "data/**",
    ]),
)

