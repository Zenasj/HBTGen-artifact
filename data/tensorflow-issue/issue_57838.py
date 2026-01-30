import tensorflow as tf
print(tf.__version__)
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(
            filters=2, kernel_size=1, padding='same',
            dtype=tf.float32, autocast=False,
        )

    @tf.function(jit_compile=True)
    def __call__(self, x):
        y = self.conv(x)
        return y



inp = {
    "x": tf.constant(1.2, shape=[1,2,2,2], dtype=tf.float32),
}
m = MyModule()

with tf.device('CPU:0'):
    out = m(**inp)
    print(f'{out}')

with tf.device('GPU:0'):
    out = m(**inp) # <--- exception!
    print(f'{out}')