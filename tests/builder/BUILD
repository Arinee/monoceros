[cc_test(name = f.split('.')[0],
        srcs = [f, "mock_vector_holder.h"],
        deps = ["//src/builder:lve_builder", "@com_google_gtest//:gtest_main",],
        copts = ["-I./src"],
        linkopts = [ "-ldl" ],
) for f in glob(["*.cc"])]
