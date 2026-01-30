jsonc
{
    "batch_size": {"low": 1, "high": 8},
    "in_channels": {"low": 16, "high": 128},
    "out_channels": {"low": 16, "high": 128},
    "height": {"low": 16, "high": 224},
    "stride": {"set": [[1, 1], [2, 2]]},
    "padding": {"set": [[0, 0]]},
    "output_padding": {"set": [[0, 0], [1, 1], [0, 1], [1, 0]]},
    "kernel_size": {"set": [[3, 3], [1, 1], [1, 3], [3, 1], [2, 2]]},
    "dilation": {"set": [[1, 1]]},
    "deterministic": {"set": [true, false]},
    "benchmark": {"set": [true, false]},
    "allow_tf32": {"set": [true, false]},
    "groups": {"set": [1, IN_CHANNELS]}
}