package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_cluster",
    srcs = glob([ "*.cc", ], exclude = ["cluster_tool.cc"]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//src/index:lve_index",
        "//src/framework:lve_framework",
        "//src/utils:lve_utils",
        "@gflags"
    ]
)
