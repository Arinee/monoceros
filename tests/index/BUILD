[cc_test(name = f.split('.')[0],
        srcs = [f, "rough_matrix.txt.h", "query_data.txt.h" ],
        deps = ["//src/index:lve_index", "//src/framework:lve_framework", "//src/utils:lve_utils", "@com_google_gtest//:gtest_main", "@gflags"],
        copts = ["-I./src"],
        linkopts = [ "-ldl" ],
) for f in glob(["*.cc"])]
