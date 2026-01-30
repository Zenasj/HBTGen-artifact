import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)
    
    # The function to calculate the gradients by myself
    def _backprop(self,y_delta,x):
        w_delta=some_function(y_delta,x)
        x_delta=some_function2(y_delta,x)
        return w_delta, x_delta