from six.moves import queue as Queue
import threading

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test

disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)  # pass if set 2


class ThreadHangTest(test.TestCase):
    """test Stage/MapStage"""

    def testStage(self):
        capacity = 3
        with ops.device(test.gpu_device_name()):
            x = array_ops.placeholder(dtypes.int32, name='x')
            stager = data_flow_ops.StagingArea([dtypes.int32, ], capacity=capacity, shapes=[[]])

        queue = Queue.Queue()
        with self.session() as sess:
            def thread_run():
                for i in range(capacity + 1):
                    sess.run(stager.put([x]), feed_dict={x: i})
                    queue.put(0)

            t = threading.Thread(target=thread_run)
            t.daemon = True
            t.start()

            try:
                for i in range(capacity + 1):
                    queue.get(timeout=1)
            except Queue.Empty:
                pass

            for i in range(capacity):
                sess.run(stager.get())

    def testMapStage(self):
        capacity = 3
        with ops.device(test.gpu_device_name()):
            x = array_ops.placeholder(dtypes.int32, name='x')
            pi = array_ops.placeholder(dtypes.int64, name='pi')
            map_stager = data_flow_ops.MapStagingArea([dtypes.int32, ], capacity=capacity, shapes=[[]])

        queue = Queue.Queue()
        with self.session() as sess:
            def thread_run():
                for i in range(capacity + 1):
                    sess.run(map_stager.put(pi, [x], [0]), feed_dict={x: i, pi: i})
                    queue.put(0)

            t = threading.Thread(target=thread_run)
            t.daemon = True
            t.start()

            try:
                for i in range(capacity + 1):
                    queue.get(timeout=1)
            except Queue.Empty:
                pass

            for i in range(capacity):
                sess.run(map_stager.get())


if __name__ == '__main__':
    test.main()