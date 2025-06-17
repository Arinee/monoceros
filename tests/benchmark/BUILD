package(
  default_visibility = ["//visibility:public"],
)

config_setting(
  name = "link_gpu",
  define_values = { "with_gpu": "true" },
  visibility = ["//visibility:public"],
)

cc_test(
  name = "benchmark_hnsw",
  srcs = glob([
    "group_hnsw/*.cc",
    "test_main.cc",
  ]),
  deps = [
    "@com_google_gtest//:gtest",
    "//src/core/algorithm:algorithm",
  ],
  data = glob([
    "data/**",
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

cc_test(
  name = "benchmark_vamana",
  srcs = glob([
    "ram_vamana/*.cc",
    "test_main.cc",
  ]),
  deps = [
    "@com_google_gtest//:gtest",
    "//src/core/algorithm:algorithm",
  ],
  data = glob([
    "data/**",
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