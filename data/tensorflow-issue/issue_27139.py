import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, out_size):
        super().__init__()
        self.rnn_layer = tf.keras.layers.GRU(out_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, X, **kwargs):
        output, state = self.rnn_layer(X)
        return output, state

class EncodingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder_layer = EncodingLayer(out_size=1)

    def infer(self, inputs):
        output, state = self.encoder_layer(inputs)
        return output


if __name__ == '__main__':
    # Comment line below for running in TF 2.0
    tf.enable_eager_execution()

    # shape == (2, 3, 2)
    inputs = tf.convert_to_tensor([
        [[1., 2.], [2., 3.], [4., 4.]],
        [[1., 2.], [2., 3.], [4., 4.]],
    ])

    model = EncodingModel()

    # Just for building the graph
    model.infer(inputs)

    print('Before saving model: ', model.trainable_weights[0].numpy().mean())
    model.save_weights('weight')

    new_model = EncodingModel()
    new_model.infer(inputs)
    new_model.load_weights('weight')
    print('Loaded model: ', new_model.trainable_weights[0].numpy().mean())