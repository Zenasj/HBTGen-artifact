def monkeypatch_tensorflow_to_work_around_eager_mode_bug():
    """Monkey-patch Tensorflow to work around an eager mode bug. The bug is, apparently, that in
    https://github.com/tensorflow/tensorflow/blob/7730bb302615aee24bbad653dcf7f7698d2dae5d/tensorflow/python/ops/ragged/ragged_conversion_ops.py#L66
    there is a comparison with a bytestring b"FIRST_DIM_SIZE", while in eager mode it's a plain str.
    """
    from tensorflow.python.eager import backprop
    original_MockOp = backprop._MockOp

    class PatchedMockOp(original_MockOp):
        def get_attr(self, attr):
            result = super().get_attr(attr)
            if attr == 'row_partition_types':
                assert isinstance(result, list)
                assert all(isinstance(item, str) for item in result)
                return [item.encode('ascii') for item in result]
            else:
                return result

    backprop._MockOp = PatchedMockOp