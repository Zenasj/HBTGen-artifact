from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import layers


class MnistModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.first_dense = layers.Dense(64, input_shape=(784,), activation='relu', name='dense_1')
    self.out = layers.Dense(10, activation='softmax', name='predictions')

  def call(self, inp):
    f_dense = self.first_dense(inp)
    s_dense = self.out(f_dense)
    return s_dense

  def input_receiver(self, inp):
    return inp

  def response_receiver(self, output):
    return output

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], name="serving")])
  def serve(self, request):
    features = tf.identity(self.input_receiver(request), name='request')
    output = self.call(features)
    response = tf.identity(self.response_receiver(output), name='response')
    return response

model = MnistModel()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

keras.experimental.export_saved_model(model, 'local_path', serving_only=True)

tf.saved_model.save(model, 'local_path', signatures={"serve": model.serve})