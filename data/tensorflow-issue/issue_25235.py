from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.probas = keras.layers.Dense(10, activation="softmax")
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, serialized):
        expected_features = {
            "image": tf.io.FixedLenFeature([28 * 28], dtype=tf.float32)
        }
        examples = tf.io.parse_example(serialized, expected_features)
        return self.probas(examples["image"])
    def call(self, inputs):
        return self.probas(inputs)

m = MyModel()
m.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
m.fit(X_train.reshape(-1, 28*28), y_train)
tf.saved_model.save(m, "my_model_test",
    signatures=m.serve.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="serialized_inputs")))