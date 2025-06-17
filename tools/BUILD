package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "local_builder",
    srcs = ["local_builder.cc", "pq_helper.h", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/builder:lve_builder",
        "//src/utils:lve_utils",
        "//src/faiss_index:lve_faiss_index",
        "@gflags",
    ],
)
cc_binary(
    name = "local_centroid_trainer",
    srcs = ["local_centroid_trainer.cc", "pq_helper.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "recall",
    srcs = ["recall.cc", "pq_helper.h", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "recall_group",
    srcs = ["group_index/recall.cc", "pq_helper.h", "group_index/txt_string_reader.h"],
    #copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/core/algorithm:algorithm",
        "@gflags",
    ],
)
cc_binary(
    name = "cat_recall",
    srcs = ["cat_recall.cc", "pq_helper.h", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "knn_search",
    srcs = ["knn_search.cc", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "bench",
    srcs = ["bench.cc", "bench_result.h", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "mips_trans_tool",
    srcs = ["mips_trans_tool.cc", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "make_groundtruth",
    srcs = ["make_groundtruth.cc", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "make_groundtruth_group",
    srcs = ["group_index/make_groundtruth.cc", "group_index/txt_string_reader.h"],
    #copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
    "//src/core/algorithm:algorithm",
        "@gflags",
    ],
)
cc_binary(
    name = "centroid_vaild_tool",
    srcs = ["centroid_vaild_tool.cc", "pq_helper.h", "txt_input_reader.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "index_show_tool",
    srcs = ["index_show_tool.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "l2norm_tool",
    srcs = ["l2norm_tool.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "print_groundtruth",
    srcs = ["print_groundtruth.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "vecs2txt",
    srcs = ["vecs2txt.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "txt2vecs",
    srcs = ["txt2vecs.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "txt2fvecs",
    srcs = ["txt2fvecs.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "lve_ms_split_data",
    srcs = ["lve_ms_split_data.cc"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
cc_binary(
    name = "dump_centroids",
    srcs = ["dump_centroids.cc", "pq_helper.h"],
    copts = [ "-I./src/", "-I./src/utils" ],
    linkopts = [ "-ldl" ],
    deps = [
        "//src/service:lve_service",
        "//src/train:lve_train",
        "//src/utils:lve_utils",
        "@gflags",
    ],
)
