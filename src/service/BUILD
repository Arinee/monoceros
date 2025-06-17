package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_service",
    srcs = glob([
        "*.cc"
    ]),
    hdrs = glob([
        "*.h",
    ]),
    alwayslink = True,
    deps = [
        "//src/index:lve_index",
    ]
)
