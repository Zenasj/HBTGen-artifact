# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred as (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Feature column expects inputs keyed by 'fc_name'
        self.fc1 = tf.feature_column.numeric_column('fc_name')
        # DenseFeatures layer uses the feature column
        self.dense_features = tf.keras.layers.DenseFeatures(self.fc1)
        # Subsequent Dense layers
        self.dense1 = tf.keras.layers.Dense(5, name='dense_feature1')
        self.dense2 = tf.keras.layers.Dense(1, name='dense_ouput')
        
    def call(self, inputs, training=False):
        # inputs: dict mapping input keys to tensors
        # The DenseFeatures layer expects keys matching feature columns ('fc_name'),
        # but the input dictionary key and the Input layer name used for the model input
        # may differ (e.g., input dict key might be 'teddy_bear')
        #
        # To work correctly with Keras .fit, the name of the Input layer should
        # correspond exactly to the input dictionary key at runtime.
        #
        # Here, we forward inputs to DenseFeatures using the expected key 'fc_name'.
        # So we extract inputs['fc_name'] if present.
        #
        # However, in real TensorFlow 2 functional API, inputs is a dict keyed by the Input layer names.
        # To reconcile that, the model Input layer must be named 'fc_name' to match the feature_column name.
        #
        # For this implementation, to keep compatibility, we assume inputs is a dict keyed by 'fc_name'.
        x = self.dense_features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Since the Functional API example uses inputs keyed by 'fc_name',
    # and DenseFeatures uses feature column with key 'fc_name',
    # we wrap the Model creation in TF Keras Functional API to properly name Input layer.
    #
    # This ensures compatibility with Keras .fit as .fit uses keyed input data and matches Input layer names.
    #
    # We'll create a Functional API model internally and wrap it inside a MyModel subclass for the required structure,
    # but to satisfy the requirement that MyModel is subclassed from tf.keras.Model,
    # and instantiable with my_model_function, we'll provide this model inside MyModel.
    #
    # Alternatively, since this inheritance and functional mix is complicated,
    # under the task requirements: produce a subclass MyModel with correct input expectations and compatible GetInput.
    #
    # So we return the subclass instance.

    # We can create a functional model here and assign it to MyModel._functional_model
    # inside __init__, but since the problem stems from name matching, it's better to create a proper subclass.

    return MyModel()

def GetInput():
    # According to the issue description:
    # - Feature column key is 'fc_name'.
    # - Input dictionary keys must match Input layer names for .fit to work.
    #
    # So inputs must be dict with key 'fc_name', tensor shape (batch_size, 1).
    #
    # We'll generate a random float tensor shaped (4, 1) with dtype float32 for batch_size=4
    # This shape matches the minimal example (batch size 4 used in the issue).
    #
    # The dict keys must match the Input layer names, which are 'fc_name' to satisfy DenseFeatures.
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 1), dtype=tf.float32)

    # Return a dict keyed by 'fc_name' matching numeric_column and Input layer name
    return {'fc_name': input_tensor}

