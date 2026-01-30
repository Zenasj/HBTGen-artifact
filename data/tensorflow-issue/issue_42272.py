import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        out = self.dense(inputs)
        return out

if __name__ == '__main__':
    data = tf.constant(tf.ones((10, 20)))

    tag = tf.constant(tf.ones((10)))
    testmodel = MyModel()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(tmodel=testmodel)
    with tf.GradientTape() as tape:
        out = testmodel(data)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tag, out))
    grad = tape.gradient(loss, testmodel.trainable_variables)
    optimizer.apply_gradients(zip(grad, testmodel.trainable_variables))

    checkpoint.save("test/model.ckpt")

if __name__ == '__main__':
    testmodel = MyModel()
    checkpoint = tf.train.Checkpoint(tmodel=testmodel)
    checkpoint.restore(tf.train.latest_checkpoint('test'))