workspace(name="mercury")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "rules_lucky",
    remote = "https://gitlab+deploy-token-80:4k2_tUkQ9QEZEirP9wz1@code.devops.xiaohongshu.com/red-monster/rules_lucky.git",
    branch = "dev.jiaolong",
)

load("@rules_lucky//bazel:init.bzl", "lucky_init")
lucky_init()

load("@rules_lucky//bazel:lucky_build_system.bzl", "lucky_repository")
lucky_repository("third_party/bazelbuild/rules_python", tag = "0.3.0")

lucky_repository("third_party/bazelbuild/bazel-skylib", name = "bazel_skylib", tag = "1.0.3")
# load("@bazel_skylib//lib:dicts.bzl", "dicts")

lucky_repository("red-monster/search_common", tag = "tag.v0.1.1")
lucky_repository("red-monster/neutron_lib", branch = "feature/pq-batch")

lucky_repository("third_party/google/googletest", tag = "release-1.10.0", name = "com_google_gtest")

lucky_repository("third_party/gflags/gflags", commit = "46f73f88b18", name = "com_github_gflags_gflags")

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

lucky_repository(
    "third_party/protocolbuffers/protobuf",
    tag = "v3.6.1.3",
    name = "com_google_protobuf",
    patch_tool = "git",
    patch_args = ["apply"],
    patches = ["@rules_lucky//patch:protobuf.bazel4.fix.patch"],
)

lucky_repository(
    "third_party/google/leveldb",
    commit = "a53934a3ae12",
    name = "com_github_google_leveldb",
    build_file = "@rules_lucky//build:leveldb.BUILD"
)


lucky_repository("rpc/brpc-red", name = "brpc", tag = "tag.v0.1.0") 
lucky_repository("red-monster/cat_client", tag = "tag.v0.1.0")

lucky_repository("third_party/abseil/abseil-cpp", name = "com_google_absl")
bind(
    name = "absl-failure_signal_handler",
    actual = "@com_google_absl//absl/debugging:failure_signal_handler",
)
bind(
    name = "absl-failure_signal_handler",
    actual = "@com_google_absl//absl/debugging:failure_signal_handler",
)

# boost library
lucky_repository("red-monster/boost", name = "com_github_nelhage_rules_boost", tag = "tag.v0.1.0")
bind(name = "boost", actual = "@com_github_nelhage_rules_boost//:boost")

lucky_repository(
    "third_party/google/glog",
    commit = "a6a166db06952",
    name = "com_github_google_glog",
    build_file = "@rules_lucky//build:glog.BUILD",
)

bind(
    name = "glog",
    actual = "@com_github_google_glog//:glog",
)

lucky_repository(
    "third_party/cameron314/concurrentqueue",
    name = "com_github_concurrentqueue",
    build_file = "@rules_lucky//build:concurrentqueue.BUILD",
    tag = "v1.0.3"
)

