package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_train",
    srcs = glob([
        "*.cc"
    ]),
    hdrs = glob([
        "*.h",
    ]),
    includes = [".."],
    deps = [
        "//src/cluster:lve_cluster",
        "//src/builder:lve_builder",
    ]
)
