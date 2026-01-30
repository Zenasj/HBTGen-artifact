os.environ["TORCHINDUCTOR_ABI_COMPATIBLE"] = "1"

dynamic_shapes={
                    "batch": [[batch_dim] for _ in example_inputs]
                },

dynamic_shapes= {"batch": [{0:batch_dim} for _ in example_inputs]},