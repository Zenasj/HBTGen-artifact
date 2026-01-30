[Singleton(), InputDim(input_dim=1), Flatten(input_dims=(InputDim(input_dim=2), InputDim(input_dim=3)))]

[InputDim(input_dim=0), InputDim(input_dim=1), Flatten(input_dims=(InputDim(input_dim=2), InputDim(input_dim=3)))]

self.assertEqual(
    view_groups([1, 1, 3, 2, 1, 1], [6, 1, 1, 1]),
    (
        Flatten((InputDim(2), InputDim(3))),
        Singleton(),
        Singleton(),
        Singleton(),
    ),
)
# since the change in the PR modifies rules and it becomes 
# (Flatten(input_dims=(InputDim(input_dim=2), InputDim(input_dim=3))), InputDim(input_dim=4), InputDim(input_dim=5), Singleton())

self.dimmap_test(
    Tensor.view,
    (randn(1, 1, 42, 1, 24, 1), -1),
    (Flatten((InputDim(2), InputDim(4))),),
)
# since the change in the PR modifies rules and it becomes 
# (Flatten(input_dims=(InputDim(input_dim=2), InputDim(input_dim=3), InputDim(input_dim=4))),)