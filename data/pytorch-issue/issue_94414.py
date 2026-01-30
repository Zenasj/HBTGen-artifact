FakeTensorMode(
    shape_env=ShapeEnv(  # all default values
        allow_scalar_outputs=True,
        allow_dynamic_output_shape_ops=True,
        assume_static_by_default=False,
    ),
)