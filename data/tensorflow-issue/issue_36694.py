# tf.random.uniform((None, 20), dtype=tf.float32) ← Inferred input shape based on issue's model Input(batch_shape=(None, 20))

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    """
    This model replicates the example used in the GitHub issue:
    - Input shape: (None, 20)
    - One Dense hidden layer with 1028 units
    - Output Dense layer with 1 unit and sigmoid activation

    The forward pass returns the output tensor.
    
    Additionally, this class includes methods to compute:
    - custom_loss2: a mean binary crossentropy loss 
    - EWC_loss: Elastic Weight Consolidation loss on weights
    
    It also provides a combined loss function similar to the issue trying to combine both.
    """

    def __init__(self):
        super().__init__()
        self.dense_1 = Dense(1028)
        self.output_2 = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        out = self.output_2(x)
        return out

    def custom_loss2(self, outputs, labels):
        # Mean binary crossentropy computed similarly as K.mean(binary_crossentropy(label2, output2))
        # Use Keras backend binary_crossentropy with default from_logits=False since output activation=sigmoid
        loss_value = K.mean(binary_crossentropy(labels, outputs))
        return loss_value

    def EWC_loss(self, new_weights, old_weights, fisher_matrix, rate):
        # Fisher matrix, old_weights, new_weights are all list of tf.Tensors (model.trainable_variables)
        sum_w = 0.0
        for v in range(len(fisher_matrix)):
            # tf.multiply performs elementwise multiplication on fisher_matrix[v] and squared difference
            sum_w += tf.reduce_sum(tf.multiply(fisher_matrix[v], tf.square(new_weights[v] - old_weights[v])))
        return sum_w * rate

def my_model_function():
    """
    Returns a fresh instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a random float32 tensor shaped (batch_size, 20),
    matching the model input, here batch_size=32 assumed for example.
    Values generated with uniform distribution in [0,1).
    """
    batch_size = 32  # assuming a typical batch size
    # According to the issue: input shape is (None, 20), no channels dimension
    return tf.random.uniform((batch_size, 20), dtype=tf.float32)

