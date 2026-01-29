# tf.random.uniform((B,)) for 'crl_avg', 'crl_long', 'crl_total', (B,47) for 'wf', (B,5) for 'cf'
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Feature columns as used in the original code: numeric columns with specified shapes
        self.feature_columns = [
            tf.feature_column.numeric_column('wf', shape=(47,)),
            tf.feature_column.numeric_column('cf', shape=(5,)),
            tf.feature_column.numeric_column('crl_avg', shape=()),
            tf.feature_column.numeric_column('crl_long', shape=()),
            tf.feature_column.numeric_column('crl_total', shape=()),
        ]
        # DenseFeatures layer to process the dictionary input of features
        self.dense_features = tf.keras.layers.DenseFeatures(self.feature_columns)
        
        # Following dense layers as per original model
        self.dense1 = tf.keras.layers.Dense(40, activation='relu')
        self.dense2 = tf.keras.layers.Dense(30, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        """
        inputs: dictionary of feature tensors with keys:
          'wf': shape (batch_size, 47)
          'cf': shape (batch_size, 5)
          'crl_avg': shape (batch_size,)
          'crl_long': shape (batch_size,)
          'crl_total': shape (batch_size,)
        """
        x = self.dense_features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    """
    Returns a dictionary of feature tensors matching input expected by MyModel.
    Batch size chosen as 4 for example.
    """
    batch_size = 4
    return {
        'wf': tf.random.uniform((batch_size, 47), dtype=tf.float32),
        'cf': tf.random.uniform((batch_size, 5), dtype=tf.float32),
        'crl_avg': tf.random.uniform((batch_size,), dtype=tf.float32),
        'crl_long': tf.random.uniform((batch_size,), dtype=tf.float32),
        'crl_total': tf.random.uniform((batch_size,), dtype=tf.float32),
    }

