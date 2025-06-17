
package(
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "use_http_lion",
    define_values = {"leo": "http_lion"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "link_tcmalloc",
    define_values = {"with_tcmalloc": "true"},
    visibility = ["//visibility:public"],
)

# load("@search_common//cppcommon/bazel:cpplint.bzl", "cpplint")

#### cc rules #####

cc_import(
    name = "jvm",
    shared_library = "@local_jdk//:jre/lib/amd64/server/libjvm.so"
)

cc_library(
    name = "lve_framework",
    deps = [
        "//src/framework:lve_framework"
    ]
)


cc_library(
    name = "lve_service",
    deps = [
        "//src/service:lve_service"
    ]
)
