package(
    default_visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "lve_faiss_index",
    srcs = glob([ "*.cc", "*.h", ]),
    hdrs = glob([
        "*.h",
    ]),
    alwayslink = True,
    deps = [
        "//src/index:lve_index",
        "//src/framework:lve_framework",
        "//src/common:lve_common"
        #"//external:faiss",
    ]
)
