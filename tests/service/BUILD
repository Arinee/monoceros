[cc_test(name = f.split('.')[0],
        srcs = [f],
        deps = ["//src/index:lve_index", "//src/framework:lve_framework", "//src/service:lve_service", "@com_google_gtest//:gtest_main", "@gflags"],
        copts = ["-I./src"],
        linkopts = [ "-ldl" ],
) for f in glob(["*.cc"])]
