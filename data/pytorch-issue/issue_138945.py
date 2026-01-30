flattened_mesh = device_mesh["dim0", "dim1", "dim2"]._flatten("dim0-2")
alias_flattened_mesh = device_mesh["dim0-2"]  # this mesh slice leads to error in current impl