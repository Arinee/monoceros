package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "mercury_framework",
    srcs = glob([
        "*.cc",
        "utility/*.cc",
        "utility/internal/*.cc",
    ]),
    hdrs = glob([
        "*.h",
        "utility/*.h",
        "utility/internal/*.h",
        "algorithm/*.h",
        "custom_distance/*.h"
    ]),
    alwayslink = True,
    deps = [ 
        "//src/core/common:mercury_common",
        "@search_common//alog:alog",
        "@search_common//half:half",
    ],
)
