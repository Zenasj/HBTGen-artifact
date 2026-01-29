# tf.random.uniform((112, 32, 32, 3), dtype=tf.float32)  # inferred from cifar10 dataset used in original code
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        # BatchNormalization layer with momentum and epsilon set as in original code
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.4639004933194679, epsilon=0.6515653837017596)
        # PReLU activation layer with alpha_initializer='zeros'
        self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros')
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer to output num_classes logits
        self.dense = tf.keras.layers.Dense(num_classes)
        
    def call(self, inputs, training=False):
        x = self.bn(inputs, training=training)
        x = self.prelu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel, with num_classes=10 as in original example
    return MyModel(num_classes=10)

def GetInput():
    # Return a random tensor input matching CIFAR-10 input shape and dtype
    # CIFAR-10 images have shape (32, 32, 3) and batch size 112
    # Using float32 dtype since input is normalized to 0-1 float in original code
    return tf.random.uniform((112, 32, 32, 3), dtype=tf.float32)

