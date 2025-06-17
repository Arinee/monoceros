package(
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "link_gpu",
    define_values = {"with_gpu": "true"},
    visibility = ["//visibility:public"],
)

#### cc rules #####

cc_library(
    name = "algorithm",
    srcs = glob([
        "*/*.cc",
        "*.cc",
        "*.h",
        "*/*.h",
    ]),
    hdrs = glob([
        "*/*.h",
        "*.h",
    ]),
    alwayslink = True,
    deps = [
        "//src/core/framework:mercury_framework",
        "//src/core/utils:mercury_utils",
        "@search_common//fslib:fslib",
        # "//:jvm",
        "@brpc//:brpc",
        "@com_github_nelhage_rules_boost//:graph",
    ]+ select({
        ":link_gpu": [
            "@neutron_lib//:neutron_lib",
        ],
        "//conditions:default": [
        ],
    }),
    copts = [
        "-fopenmp"
    ] + select({
        ":link_gpu": [
            "-DENABLE_GPU_IN_MERCURY_"
        ],
        "//conditions:default": [
        ],
    }),
    linkopts = [
        "-fopenmp"
    ]
)
