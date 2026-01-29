# tf.random.uniform((B, D), dtype=tf.float32)  # Input shape inferred as (batch_size, num_features)
import tensorflow as tf
from tensorflow.keras import layers, regularizers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assume input dimension is known from one_hot_encoder + numeric features count
        # For this isolated model, we'll parametrize input_dim as an example.
        # Because categorical + numeric features number is not explicitly provided,
        # let's define placeholders and comments:
        #
        # NOTE: Replace `input_dim` with actual number of features derived from preprocessing pipeline.
        self.input_dim = 100  # Placeholder, replace as needed
        
        self.dense1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)

        self.dense2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)

        self.output_layer = layers.Dense(3, activation='softmax')  # 3 classes: push, miss, hit

    def call(self, inputs, training=False):
        # inputs: Tensor, shape (batch_size, input_dim)
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)

        output = self.output_layer(x)
        return output


def my_model_function():
    """
    Return an instance of MyModel.
    Note: The caller must handle input shape matching, e.g. inputs must be shape (batch_size, input_dim).
    """
    return MyModel()


def GetInput():
    """
    Generate a random tensor input compatible with MyModel.
    Because actual input dimension is not explicitly given, we infer based on the preprocessing:

    - OneHotEncoder transforms categorical columns into a vector.
    - StandardScaler normalizes numeric features.
    - These are concatenated horizontally to create the final feature vector.

    For this example, let's assume total features is 100 as placeholder.
    In practice, set this to number of one-hot categories + number of numeric features.
    """
    batch_size = 32  # Typical batch size for example
    input_dim = 100  # Must match MyModel's `input_dim`

    # Generate random float32 tensor with uniform distribution [0, 1)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

