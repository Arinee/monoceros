package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_index",
    srcs = glob([ "*.cc", "*.h", ]),
    hdrs = glob([
        "*.h",
    ]),
    alwayslink = True,
    deps = [
        "//src/utils:lve_utils",
        "//src/framework:lve_framework",
        "//src/common:lve_common",
    ]
)
