# tf.random.uniform((32, 1), dtype=tf.string) ← The input feature 'thal' is a string tensor of shape (batch_size, 1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define categorical column with vocabulary list
        thal = tf.feature_column.categorical_column_with_vocabulary_list(
            key='thal',
            vocabulary_list=['fixed', 'normal', 'reversible'],
            dtype=tf.string)

        # Indicator column converts categorical column to one-hot encoding
        thal_one_hot = tf.feature_column.indicator_column(thal)

        # DenseFeatures layer to process feature columns
        self.feature_layer = tf.keras.layers.DenseFeatures([thal_one_hot])

        # Define the Dense layers of the model
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')

        # Final output layer with sigmoid activation (binary classification)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs is a dictionary of feature name to tensors, e.g. {'thal': <tf.Tensor>}
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dictionary input matching the model's expectation with random 'thal' strings
    # Vocabulary: ['fixed', 'normal', 'reversible']
    import numpy as np

    batch_size = 32
    vocab = np.array(['fixed', 'normal', 'reversible'], dtype=object)

    # Randomly sample strings from vocabulary for batch_size
    sampled = np.random.choice(vocab, size=(batch_size, 1))

    # Create input dictionary with 'thal' tensor of shape (batch_size, 1), dtype tf.string
    input_dict = {
        'thal': tf.constant(sampled)
    }
    return input_dict

