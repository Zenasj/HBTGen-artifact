from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf

class TestModel(tf.keras.Model):

    def __init__(self, N):
        super(TestModel, self).__init__()
        self.conv_first = tf.keras.layers.Conv2D(4, (3, 3))
        self.nframe = N

    def __call__(self, x):

        aligned_fea = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        def cond(i, N, fea_col):
            return i < N

        def body(i, N, fea_col):
            fea_col = fea_col.write(i, x)
            i = tf.add(i, 1)
            return i, N, fea_col

        _, _, aligned_fea = tf.while_loop(cond, body, [0, self.nframe, aligned_fea])

        tf.print("aliged_fea shape:", aligned_fea.size())

        t = aligned_fea.stack()
        tf.print("t shape:", t.shape)

        # without these two lines of reshaping the stacked tensor coercively, the t will have no first dimension
        tt = tf.reshape(t,[self.nframe, 8, 4, 6,3])
        tf.print("tt shape:", tt.shape)

        return t

@tf.function
def foo(tm):
    x = tf.ones([8,4,6,3], dtype=tf.float32)
    output = tm(x)

nframe = 10
tm = TestModel(nframe)
foo(tm)