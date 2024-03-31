cc_library(
    name = "micrograd",
    srcs = ["micrograd.cc"],
    hdrs = ["micrograd.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:any_invocable",
    ],
)

cc_test(
        name = "micrograd_test",
        srcs = ["micrograd_test.cc"],
        deps = [
                ":micrograd",
                "@googletest//:gtest",
                "@googletest//:gtest_main",
        ],
)
