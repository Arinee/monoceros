package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_framework",
    srcs = glob([
        "*.cc",
        "utility/*.cc",
        "utility/internal/*.cc",
    ]),
    hdrs = glob([
        "*.h",
        "utility/*.h",
        "utility/internal/*.h",
        "algorithm/*.h"
    ]),
    alwayslink = True,
    deps = [ 
        "//src/common:lve_common",
    ],
)
