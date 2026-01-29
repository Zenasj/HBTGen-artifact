# tf.random.uniform((50, 28, 28, 1), dtype=tf.float32) ← Input batch size 50, height 28, width 28, channels 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Convolution Layer 1: 3x3 kernel, 8 filters, stride 1, padding same
        self.conv1_kernel = tf.Variable(
            tf.random.truncated_normal([3, 3, 1, 8], stddev=0.1),
            name="filter_1"
        )
        # Dense layer parameters
        # We'll infer L1 dynamically on first call since shape depends on batch
        self.W1 = None
        self.b1 = None
        self.L2 = 500
        self.W2 = tf.Variable(
            tf.random.normal([self.L2, 10], stddev=0.01, mean=0.0),
            name="W2"
        )
        self.b2 = tf.Variable(
            tf.random.normal([10], stddev=0.01, mean=0.0),
            name="b2"
        )
        self._initialized_dense1 = False

    def call(self, inputs):
        # inputs shape: (50, 28, 28, 1)
        # Convolution
        conv_1 = tf.nn.conv2d(inputs, self.conv1_kernel, strides=[1,1,1,1], padding="SAME")
        
        batch_size = tf.shape(conv_1)[0]
        # Flatten conv_1 for Dense layer:
        reshaped = tf.reshape(conv_1, [batch_size, -1])
        L1 = reshaped.shape[1]  # features dimension after conv

        # Lazy initialize W1 and b1 since input dim depends on actual image size
        if not self._initialized_dense1:
            self.W1 = tf.Variable(
                tf.random.normal([L1, self.L2], stddev=0.01, mean=0.0),
                name="W1"
            )
            self.b1 = tf.Variable(
                tf.random.normal([self.L2], stddev=0.01, mean=0.0),
                name="b1"
            )
            self._initialized_dense1 = True
        
        # Dense layer 1 with ReLU
        relu_1 = tf.nn.relu(tf.matmul(reshaped, self.W1) + self.b1)
        
        # Dense layer 2 with ReLU (as per original code, though logits generally are linear)
        logits = tf.nn.relu(tf.matmul(relu_1, self.W2) + self.b2)
        
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape [50, 28, 28, 1], dtype float32
    # Matches the batch size and input expected in original model
    return tf.random.uniform(shape=[50, 28, 28, 1], dtype=tf.float32)

