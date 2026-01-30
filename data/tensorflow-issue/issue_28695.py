import random

# Graph mode
import tensorflow as tf

x = tf.random.normal([5], seed=1)

with tf.Session() as sess:
    for i in range(3):
        print(sess.run(x))

# Output
# [-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]
# [-0.36332107 -0.07205155 -0.5527937   0.10289733 -0.39558855]
# [-0.666205   -0.416783    1.8211031   0.680353   -0.26143482]

# Eager mode
import tensorflow as tf

tf.enable_eager_execution()
x = tf.random.normal([5], seed=1)

for i in range(3):
    print(x.numpy())

# Output
# [-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]
# [-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]
# [-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]

import tensorflow as tf
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class DownSampleTest(tf.test.TestCase):
    def testRand(self):
        x = tf.random.normal([5], seed=1)
        y1 = self.evaluate(x)
        y2 = self.evaluate(x)
        # ... check that y1 not equals y2