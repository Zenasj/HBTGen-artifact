TORCH_CHECK(options.device() == out.device(),
    "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");