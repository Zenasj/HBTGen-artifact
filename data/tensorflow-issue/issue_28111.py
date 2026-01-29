# tf.random.uniform({'c1': (B, 1), 'c2': (B, 1)}, dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define feature columns for two numeric features 'c1' and 'c2'
        self.feature_columns = [tf.feature_column.numeric_column(key) for key in ['c1', 'c2']]
        # DenseFeatures layer consumes a dict input of features and produces a dense tensor
        self.feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns)
        # A dense layer to perform further processing after feature layer
        self.dense_layer = tf.keras.layers.Dense(8)
        
    def call(self, inputs):
        """
        Expects inputs as a dict of tensors corresponding to the feature columns.
        Each tensor shape should be (batch_size, 1), matching the numeric feature.
        """
        x = self.feature_layer(inputs)  # transforms dict input feature columns to dense tensor
        output = self.dense_layer(x)
        return output

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a dictionary input compatible with MyModel.
    Each feature key maps to a tensor of shape (batch_size, 1).
    Here batch_size is chosen as 1 for simplicity.
    The tensor values are random floats.
    """
    batch_size = 1
    input_dict = {
        'c1': tf.random.uniform(shape=(batch_size, 1), dtype=tf.float32),
        'c2': tf.random.uniform(shape=(batch_size, 1), dtype=tf.float32)
    }
    return input_dict

