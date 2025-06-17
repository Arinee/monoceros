[cc_test(name = f.split('.')[0],
        srcs = [f],
        deps = ["//src/common:lve_common", "//src/framework:lve_framework", "//src/utils:lve_utils", "@com_google_gtest//:gtest_main", "@gflags"],
        copts = ["-I./src"],
        linkopts = [ "-ldl" ],
) for f in glob(["*.cc"])]
