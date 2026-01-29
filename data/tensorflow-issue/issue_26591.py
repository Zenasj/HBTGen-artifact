# tf.random.uniform((B,)) for each feature key 'age' and 'height' (scalar float inputs)

import tensorflow as tf

# We infer the input shape from the example: model used features "age" and "height", each is scalar float.
# So input is a dict of keys to tensors of shape (batch_size, 1) or just (batch_size,).
# For simplicity, we generate tensors of shape (batch_size,) as inputs for each scalar float feature.


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define numeric feature columns for 'age' and 'height', emulating the example setup
        self.age_column = tf.feature_column.numeric_column("age")
        self.height_column = tf.feature_column.numeric_column("height")
        self.feature_columns = [self.age_column, self.height_column]

        # DenseFeatures layer for processing feature columns into dense tensors
        self.dense_features = tf.keras.layers.DenseFeatures(self.feature_columns)

        # Following the original example: a Dense layer with single output (e.g. weight regression)
        self.dense_output = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """
        inputs: dict mapping feature keys to tensors, each tensor shape (batch, ),
                e.g. {"age": tf.Tensor, "height": tf.Tensor}

        Note:
        DenseFeatures expects inputs in dict form and converts them using feature columns.
        Outputs a dense tensor which we pass through one dense layer.

        We also document the hack/fix: 
        The original reported error was "'str' object has no attribute 'shape'" or 
        "'DenseFeatures' object has no attribute 'shape'". Here, to avoid this,
        we ensure the input dict keys map to tensors, not strings.
        """
        # Run DenseFeatures layer to convert dict of feature tensors into a dense representation
        features = self.dense_features(inputs)

        # Pass features through final dense output layer
        output = self.dense_output(features)
        return output


def my_model_function():
    """
    Returns an instance of MyModel.
    This includes the DenseFeatures layer over numeric columns "age" and "height",
    followed by one Dense layer outputting a scalar.
    """
    model = MyModel()
    # We can optionally build the model by providing dummy input shape
    dummy_input = GetInput()
    _ = model(dummy_input)  # run once to build weights
    return model


def GetInput():
    """
    Returns a dictionary matching the input expected by MyModel,
    i.e. a dict with keys "age" and "height" containing float32 tensors
    of shape (batch_size,). We'll pick batch_size=32 as reasonable default here.

    This matches how DenseFeatures expects inputs: dict mapping string feature names to tensors.

    Assumptions:
    - Each feature is a scalar float.
    - This tensor represents a batch of feature values.
    """
    batch_size = 32
    inputs = {
        "age": tf.random.uniform(shape=(batch_size,), minval=20.0, maxval=70.0, dtype=tf.float32),
        "height": tf.random.uniform(shape=(batch_size,), minval=150.0, maxval=190.0, dtype=tf.float32)
    }
    return inputs

