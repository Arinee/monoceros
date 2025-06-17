package(
    default_visibility = ["//visibility:public"],
)

cc_test(
  name = "all_test",
  srcs = glob([
       "test/common.h",
       "test/*.cc",
       "test_main.cc"
       ]),
  deps = [
          "@com_google_gtest//:gtest", "//src/engine/redindex:RedIndex", "@search_common//fslib:fslib", "//:jvm",
         ],
  data = glob([
        "group_ivf_test_data/**",
        "test_data/**",
        "log.conf",
        ]),
)
