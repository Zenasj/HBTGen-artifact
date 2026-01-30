import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.platform import test

disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)  # pass if set 2


class ThreadHangTest(test_util.TensorFlowTestCase):
    """reproduce thread hang when setting inter_op=1."""

    def testLoweringDisabledWithSingleThreadedExecutorContext(self):
        with self.session() as sess:
            @function.defun
            def _add_cond(y):
                return cond_v2.cond_v2(constant_op.constant(True, name="pred"),
                                       lambda: y,
                                       lambda: y + 1)

            x = array_ops.placeholder(shape=None, dtype=dtypes.float32)
            with context.function_executor_type("SINGLE_THREADED_EXECUTOR"):
                out_cond = _add_cond(x)
            sess.run(out_cond, feed_dict={x: 1.0})
            
 
if __name__ == '__main__':
    test.main()