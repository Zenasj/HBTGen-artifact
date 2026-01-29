# tf.random.uniform((B, ?), dtype=tf.float32) ← Input shape inferred as (batch_size, number_of_features); exact shape depends on feature columns number

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import DenseFeatures
from tensorflow.feature_column import numeric_column, bucketized_column

class MyModel(tf.keras.Model):
    def __init__(self, feature_names, bucket_boundaries=None, **kwargs):
        """
        feature_names: List of strings, column names of numeric features.
        bucket_boundaries: List of bucket boundaries for bucketized columns (same for all features here for simplicity).
        
        This model replicates the original issue's model:
        - Uses DenseFeatures with bucketized numeric columns.
        - Then a stack of Dense layers with relu activations.
        - Output layer with 3 sigmoid units (i.e. multi-label classification).
        
        Note: We explicitly keep feature_columns as attributes to ensure trackability and saving works properly.
        """
        super().__init__(**kwargs)
        # Build feature columns based on inputs
        self.feature_columns = []
        for header in feature_names:
            num_col = numeric_column(header, dtype=tf.float32)
            if bucket_boundaries is not None:
                buck_col = bucketized_column(num_col, boundaries=bucket_boundaries)
                self.feature_columns.append(buck_col)
            else:
                self.feature_columns.append(num_col)

        # DenseFeatures layer wraps feature_columns
        self.feature_layer = DenseFeatures(self.feature_columns)
        
        # Dense layers as per original model structure
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(3, activation='sigmoid')  # Output 3 units, sigmoids

    def call(self, inputs):
        """
        Input is expected to be a dictionary mapping feature names to float tensors.
        This matches the expected input for DenseFeatures with feature columns.
        """
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.output_layer(x)
        return out


def my_model_function():
    # We'll create a hypothetical list of feature names and bucket boundaries based on the example:
    feature_names = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4']  # Example feature names
    boundaries = [-89, -70, -65, -60, -55, -50, -40, -30, -20]
    model = MyModel(feature_names=feature_names, bucket_boundaries=boundaries)
    # Build the model once with dummy input so weights are created (necessary for saving in some TF versions)
    dummy_input = GetInput()
    _ = model(dummy_input)
    return model


def GetInput():
    """
    Return a dictionary of feature_name -> value tensors matching the input expected by MyModel.
    Each feature tensor is a tf.Tensor of shape (batch_size,), float32 dtype.
    We assume batch_size=4 here for an example.
    Feature values are generated uniformly between -100 and 0 to cover the bucket boundaries.
    """
    batch_size = 4
    feature_names = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4']
    input_dict = {}
    for feat in feature_names:
        # Uniform values in range [-100, 0] float32
        input_dict[feat] = tf.random.uniform(shape=(batch_size,), minval=-100.0, maxval=0.0, dtype=tf.float32)
    return input_dict

