import tensorflow as tf
from tensorflow import keras

class DigitCapsuleLayer(Layer):
    # creating a layer class in keras
    def __init__(self, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        
    
    def build(self, input_shape): 
        # initialize weight matrix for each capsule in lower layer
        self.W = self.add_weight(shape = [2, 6*6*6*32, 16, 8], name = 'weights')
        self.built = True
    
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, 2, 1, 1])
        # matrix multiplication b/w previous layer output and weight matrix
        inputs = tf.map_fn(lambda x: tf.keras.backend.batch_dot(x, self.W, [2, 3]), elems=inputs)
        b = tf.zeros(shape = [tf.shape(inputs)[0], 2, 6*6*6*32])
        
        # routing algorithm with updating coupling coefficient c, using scalar product b/w input capsule and output capsule
        for i in range(3-1):
            c = tf.nn.softmax(b, dim=1)
            s = tf.keras.backend.batch_dot(c, inputs, [2, 2])
            v = squash(s)
            b = b + tf.keras.backend.batch_dot(v, inputs, [2,3])
            
        return v 
    def compute_output_shape(self, input_shape):
        return tuple([None, 10, 16])

def output_layer(inputs):
    return tf.sqrt(tf.sum(tf.square(inputs), -1) + tf.epsilon())

digit_caps = DigitCapsuleLayer()(squashed_output)
outputs = Lambda(output_layer)(digit_caps)