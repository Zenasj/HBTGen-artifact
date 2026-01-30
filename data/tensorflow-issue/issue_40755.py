import random

import tensorflow as tf

class Test:
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    @tf.function
    def compute(self, tensor):
        # some dummy computation with a tensor
        print('python.print ===> tracing compute ... ')

        res = 100*tensor
        res = tf.signal.rfft(res)  # perform some computationally heavy task

        return res

    def test_on_ds(self, N):
        # apply self.compute on a dataset made of N random tensors of length 10000

        tensors = tf.random.uniform(shape=[N, 10000], dtype=tf.float32)
        ds      = tf.data.Dataset.from_tensor_slices(tensors)

        t1 = tf.timestamp()
        count  = tf.constant(0, dtype=tf.int32)

        ds1 = ds.map(self.compute)
        for tensor in ds1:
            count += 1

        t2 = tf.timestamp()
        tf.print('time elapsed=', t2 - t1 )

        return t2 - t1

    def test_on_tensors(self, N):
        # apply self.compute on a N random tensors of length 10000

        tensors = tf.random.uniform(shape=[N, 10000], dtype=tf.float32)

        t1 = tf.timestamp()
        count  = tf.constant(0, dtype=tf.int32)

        for i in tf.range(N):
            x = self.compute( tensors[i] )
            count += 1

        t2 = tf.timestamp()
        tf.print('time elapsed=', t2 - t1 )

        return t2 - t1

import tensorflow as tf
import test

T = test.Test()

a = []
for _ in range(10):
    a.append( T.test_on_ds(100) )

# the result of tf.print

# time elapsed= 1.0664479732513428
# time elapsed= 1.063709020614624
# time elapsed= 1.0634510517120361
# time elapsed= 1.0631310939788818
# time elapsed= 1.0632259845733643
# time elapsed= 1.0634918212890625
# time elapsed= 1.0678679943084717
# time elapsed= 1.0707681179046631
# time elapsed= 1.0651659965515137
# time elapsed= 1.0619449615478516

# average runtime ~ 1.06 sec

b = [] 
for _ in range(10):
    b.append(T.test_on_tensors(100))

# the result of tf.print

# time elapsed= 0.04901123046875
# time elapsed= 0.04906916618347168
# time elapsed= 0.049163103103637695
# time elapsed= 0.049206018447875977
# time elapsed= 0.05437779426574707
# time elapsed= 0.053722143173217773
# time elapsed= 0.053966999053955078
# time elapsed= 0.053440093994140625
# time elapsed= 0.053852081298828125
# time elapsed= 0.054525852203369141

# average runtime ~ 0.05 sec

import tensorflow as tf
import test

T = test.Test()

a = []
for _ in range(10):
    a.append( T.test_on_ds(100) )

T = Test()

a = []
for _ in range(10):
    a.append( T.test_on_ds(100) )

a = []
for _ in range(10):
    a.append( T.test_on_ds(100) )

#python.print ===> tracing compute ... 
#time elapsed= 2.00871205329895
#time elapsed= 1.8355748653411865
#time elapsed= 1.836921215057373
#time elapsed= 1.8343219757080078
#time elapsed= 1.8295071125030518
#time elapsed= 1.8368699550628662
#time elapsed= 1.8276548385620117
#time elapsed= 1.8265008926391602
#time elapsed= 1.8188469409942627
#time elapsed= 1.8279800415039062

b = [] 
for _ in range(10):
    b.append(T.test_on_tensors(100))

#python.print ===> tracing compute ... 
#time elapsed= 1.8935267925262451
#time elapsed= 1.8384511470794678
#time elapsed= 1.8555481433868408
#time elapsed= 1.8480641841888428
#time elapsed= 1.8555669784545898
#time elapsed= 1.8445429801940918
#time elapsed= 1.8436379432678223
#time elapsed= 1.8572680950164795
#time elapsed= 1.8487811088562012
#time elapsed= 1.8445978164672852