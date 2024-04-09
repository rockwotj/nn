cc_library(
    name = "micrograd",
    srcs = ["micrograd.cc", "nn.cc"],
    hdrs = ["micrograd.h", "nn.h"],
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
        "@pytorch//:libtorch",
    ],
)

cc_binary(
    name = "lecture_demo",
    srcs = ["lecture_demo.cc"],
    deps = [
        ":micrograd",
    ],
)

cc_binary(
    name = "nn_demo",
    srcs = ["nn_demo.cc"],
    deps = [
        ":micrograd",
        "@plot//:plot",
        "@nlohmann_json//:json",
    ],
)
