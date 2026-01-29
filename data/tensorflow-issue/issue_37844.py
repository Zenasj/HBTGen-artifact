# tf.random.uniform((B, ) with a string feature dictionary input, shape unknown but categorical feature "category" expected shape (batch,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the categorical feature column with vocabulary list "A", "B", "C"
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key="category", vocabulary_list=["A", "B", "C"]
        )
        # Convert categorical_column to indicator column (one-hot encoding)
        indicator_column = tf.feature_column.indicator_column(cat_column)
        # DenseFeatures layer to process feature columns from dictionary input
        self.feature_layer = tf.keras.layers.DenseFeatures([indicator_column])
        # Define subsequent dense layers as in original model
        self.dense1 = tf.keras.layers.Dense(10, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, inputs, training=False):
        # inputs: dictionary with key "category" mapping to a string tensor shape (batch_size,)
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel model for direct use
    return MyModel()

def GetInput():
    # Generate a dictionary input with key "category" and batch size 2 as in example:
    # Since the model expects {"category": tf.Tensor(shape=(batch,), dtype=tf.string)}
    # We generate a batch of 2 strings from the vocabulary "A", "B", "C"
    # For tf.keras.layers.DenseFeatures with categorical_column_with_vocabulary_list, the input shape is (batch,)
    # The example used batch size 2
    categories = tf.constant(["A", "B"], dtype=tf.string)
    inputs = {"category": categories}
    return inputs

