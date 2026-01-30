PythonRefInfo(
        "_refs.chunk",
        torch_opinfo_name="chunk",
        skips=(
            # RuntimeError: Tracing expected 3 arguments but got 2 concrete arguments
            DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor'),
        ),
    ),