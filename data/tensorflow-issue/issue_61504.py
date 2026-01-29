# tf.random.normal([1, 10000]) <- input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Weights for pre_process_network: shape [10000, 10000]
        self.pre_w1 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)
        self.pre_w2 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)
        self.pre_w3 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)

        # Weights for mid_process_network_async: shapes [10000, 100], [100, 100], [100, 10000]
        self.mid_w1 = tf.Variable(tf.random.normal([10000, 100]), trainable=True)
        self.mid_w2 = tf.Variable(tf.random.normal([100, 100]), trainable=True)
        self.mid_w3 = tf.Variable(tf.random.normal([100, 10000]), trainable=True)

        # Weights for post_process_network: shape [10000, 10000]
        self.post_w1 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)
        self.post_w2 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)
        self.post_w3 = tf.Variable(tf.random.normal([10000, 10000]), trainable=True)

    def pre_process_network(self, input_tensor):
        # 3 matmuls with pre-process weights
        x = tf.matmul(input_tensor, self.pre_w1)
        x = tf.matmul(x, self.pre_w2)
        x = tf.matmul(x, self.pre_w3)
        return x

    def mid_process_network_async(self, input_tensor):
        # Adds 100 to input_tensor as per description before matmuls
        x = input_tensor + 100.0
        # 3 matmuls with mid-process weights
        x = tf.matmul(x, self.mid_w1)
        x = tf.matmul(x, self.mid_w2)
        x = tf.matmul(x, self.mid_w3)
        return x

    def post_process_network(self, input_tensor):
        # 3 matmuls with post-process weights
        x = tf.matmul(input_tensor, self.post_w1)
        x = tf.matmul(x, self.post_w2)
        x = tf.matmul(x, self.post_w3)
        return x

    @tf.function(jit_compile=True)
    def call(self, input_tensor):
        # Run all three sub-networks sequentially:
        x = self.pre_process_network(input_tensor)
        x = self.mid_process_network_async(x)
        x = self.post_process_network(x)
        return x

def my_model_function():
    # Instantiate model with initialized variables
    return MyModel()

def GetInput():
    # According to the original code inputs had shape [1, 10000]
    # dtype float32 assumed default for tf.random.normal
    return tf.random.normal((1, 10000))

