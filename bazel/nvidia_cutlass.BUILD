load("@spulib//bazel:spu.bzl", "spu_cc_library")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cc_library(
    name = "cutlass",
    srcs = [],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
