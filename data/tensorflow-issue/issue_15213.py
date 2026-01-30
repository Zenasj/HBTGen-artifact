tf_library(
    name = "test_graph_tfmatmul",
    testonly = 1,
    config = "test_graph_tfmatmul.config.pbtxt",
    cpp_class = "foo::bar::MatMulComp",
    graph = "test_graph_tfmatmul.pb",
    tags = [
        "manual",
    ],
    tfcompile_flags = "--target_triple=x86_64-pc-windows"
)