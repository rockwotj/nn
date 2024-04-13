load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

# The object files don't have the suffix in the linked binaries, so make sure to remove them here so they can be properly linked.
copy_file(
    name = "rename_libnvtc_so",
    src = "lib/libnvrtc-b51b459d.so.12",
    out = "lib/libnvrtc.so.12",
)

copy_file(
    name = "rename_libnvtc_builtins_so",
    src = "lib/libnvrtc-builtins-6c5639ce.so.12.1",
    out = "lib/libnvrtc-builtins.so.12.1",
)

copy_file(
    name = "rename_libnv_tools_ext_so",
    src = "lib/libnvToolsExt-847d78f2.so.1",
    out = "lib/libnvToolsExt.so.1",
)

copy_file(
    name = "rename_libgomp_so",
    src = "lib/libgomp-98b21ff3.so.1",
    out = "lib/libgomp.so.1",
)

copy_file(
    name = "rename_libcublas_so",
    src = "lib/libcublas-37d11411.so.12",
    out = "lib/libcublas.so.12",
)

copy_file(
    name = "rename_libcublaslt_so",
    src = "lib/libcublasLt-f97bfc2c.so.12",
    out = "lib/libcublasLt.so.12",
)

copy_file(
    name = "rename_libcudart_so",
    src = "lib/libcudart-9335f6a2.so.12",
    out = "lib/libcudart.so.12",
)

config_setting(
    name = "is_opt",
    values = {
        "compilation_mode": "opt",
    },
)

cc_library(
    name = "libtorch",
    srcs =
        select({
            ":is_opt": glob(
                ["lib/lib*.so*"],
                exclude = [
                    "lib/libtorch_python.so",
                    "lib/libnnapi_backend.so",
                    "lib/libnvrtc-*.so.12*",
                ],
            ) + [
                ":rename_libnvtc_builtins_so",
                ":rename_libnvtc_so",
            ],
            "//conditions:default": glob(
                ["lib/lib*.so*"],
                exclude = [
                    "lib/libnvrtc-*.so.12*",
                    "lib/libgomp-*.so.1",
                    "lib/libnvToolsExt-*.so.1",
                    "lib/libcublas-*.so.12",
                    "lib/libcublasLt-*.so.12",
                    "lib/libcudart-*.so.12",
                    "lib/libtorch_python.so",
                    "lib/libnnapi_backend.so",
                ],
            ) + [
                ":rename_libcublas_so",
                ":rename_libcublaslt_so",
                ":rename_libcudart_so",
                ":rename_libgomp_so",
                ":rename_libnv_tools_ext_so",
                ":rename_libnvtc_builtins_so",
                ":rename_libnvtc_so",
            ],
        }),
    hdrs = glob([
        "include/ATen/**",
        "include/c10/**",
        "include/caffe2/**",
        "include/torch/**",
        "include/torch/csrc/**",
        "include/torch/csrc/jit/**",
        "include/torch/csrc/api/include/**",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    visibility = ["//visibility:public"],
)
