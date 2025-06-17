package(
  default_visibility = ["//visibility:public"],
)

config_setting(
  name = "link_gpu",
  define_values = { "with_gpu": "true" },
  visibility = ["//visibility:public"],
)

cc_test(
  name = "all_test",
  srcs = glob([
    "group_ivf/*.cc",
    "group_ivf/common.*",
    "group_hnsw/*.cc",
    "group_ivf_pq/*.cc",
    "gpu_group_ivf/*.cc",
    "gpu_group_ivf_pq/*.cc",
    "gpu_ivf_rpq/*.cc",
    "vamana/*.cc",
    "ivf_fast_scan/*.cc",
    "ivf_rpq/*.cc",
    "utils/*.cc",
    "utils/vamana/*.cc",
    "test_main.cc",
  ],
    exclude = ["group_hnsw/group_hnsw_local_test.cc",]
  ),
  deps = [
    "@com_google_gtest//:gtest",
    "//src/core/algorithm:algorithm",
  ],
  data = glob([
    "group_ivf/test_data/**",
    "group_ivf/test_data_mips/**",
    "group_ivf/test_data_two_stage/**",
    "group_ivf_pq/test_data/**",
    "group_ivf_pq/test_data_residual/**",
    "group_ivf_pq/test_data_two_stage/**",
    "group_ivf_pq/test_data_two_stage_residual/**",
    "group_ivf_pq/test_data_1/**",
    "ivf_rpq/test_data/**",
    "ivf_rpq/test_data_residual/**",
    "ivf_rpq/test_data_two_stage_residual/**",
    "vamana/test_data/**",
    "ivf_fast_scan/test_data/**",
    "ivf_fast_scan/test_data_1/**",
    "vamana/test_data_shoe/**",
    "vamana/test_data_shoe_5m/**",
    "log.conf",
  ]),
  linkopts = [
    '-ldl',
    "-fopenmp"
  ],
  copts = [
    "-fopenmp"
  ] + select({
    ":link_gpu": [
      "-DENABLE_GPU_IN_MERCURY_"
    ],
    "//conditions:default": [
    ],
  })
)