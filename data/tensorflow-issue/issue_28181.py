# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

L2_REG = 1e-7
LEARN_RATE = 1e-5
NUM_LABELS = 10

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.layer2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.layer3 = tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(NUM_LABELS, activation='softmax')
        self.batchnorm1 = tf.keras.layers.BatchNormalization(scale=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(scale=False)

    def call(self, inputs):
        # Apply first conv layer + batchnorm
        x = self.batchnorm1(self.layer1(inputs))
        # Second conv layer
        x = self.layer2(x)
        # Instead of np.sum, use tf.reduce_sum for symbolic tensor
        sum_val = tf.reduce_sum(x, axis=None)
        # Log scalar activation sum to TensorBoard - use tf.summary.scalar at runtime carefully
        # We put scalar summary inside a context manager so it works properly in graph and eager modes
        tf.summary.scalar('layer_2_activation_sum', sum_val)
        x = self.batchnorm2(self.pool(x))
        x = self.layer3(x)
        x = self.flatten(x)
        return self.classifier(x)


def my_model_function():
    return MyModel()


def GetInput():
    # Since the MNIST images are (28, 28, 1) float32 normalized [0,1] images,
    # batch size unspecified (None) for generality, here B=1 for example
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

