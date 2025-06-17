package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "mercury_utils",
    srcs = glob([
        "*.cc",
        "*/*.cpp",
        "*/*.h",
        "*/*.cc",
        "*/*/*.h"
    ]),
    hdrs = glob([
        "*.h",
        "*/*.h"
    ]),
    deps = [
        "//src/core/framework:mercury_framework",
        "@search_common//putil:putil",
        "@brpc//:brpc",
        "@cat_client//:cat",
        "@com_google_absl//absl/synchronization:synchronization",
        "@com_github_google_leveldb//:leveldb"
    ],
    linkopts = [
        '-laio',
    ],
    alwayslink = True,
    visibility = ["//visibility:public"]
)
