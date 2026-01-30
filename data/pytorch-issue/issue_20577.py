op = core.CreateOperator(
    "ATen",
    ["X2", "X1"],
    ["Y"],
    operator="grid_sampler"
)