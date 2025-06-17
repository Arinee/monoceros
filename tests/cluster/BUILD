[cc_test(name = f.split('.')[0],
        srcs = [f],
        deps = ["//src/cluster:lve_cluster", "@com_google_gtest//:gtest_main",],
        copts = ["-I./src"],
        linkopts = [ "-ldl" ],
) for f in glob(["*.cc"])]
