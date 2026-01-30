import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="foo",
    sources=["foo.cpp"],
    extra_include_paths=["foo/this is/a path/with spaces/"],
    verbose=True,
)