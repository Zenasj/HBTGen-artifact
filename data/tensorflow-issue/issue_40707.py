# tf.random.uniform((B,), dtype=tf.string) ← The inputs are string scalar tensors representing categorical fields

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, feature_columns):
        super().__init__()
        # DenseFeatures Layer using provided feature columns
        self.dense_features = tf.keras.layers.DenseFeatures(feature_columns)
        
    def call(self, inputs):
        # inputs is expected to be a dict mapping feature names to tensors
        # For demonstration, extract from dict for DenseFeatures
        # Note: In the reported issue, DenseFeatures silently ignores missing keys.
        # To emulate expected behavior, we raise KeyError for missing keys before calling DenseFeatures.
        for key in self.dense_features.feature_column_names:
            if key not in inputs:
                raise KeyError(f"Input dict missing key required by DenseFeatures: '{key}'")
        
        # Obtain DenseFeatures output
        features_output = self.dense_features(inputs)
        
        # Also return the raw inputs concatenated for comparison, to reflect usage similar to dummy_model_1
        # Here we just concatenate the string tensors after conversion to numerical hash for succinct numeric output
        # This is a proxy for returning inputs directly, since Keras models cannot output string tensors directly during training
        def string_to_hash(t):
            # simple hash for string tensor to numeric tensor for demonstration
            return tf.strings.to_hash_bucket_fast(t, 1000)
        
        # Convert all inputs to hashed numeric form and concatenate
        input_hashes = [tf.squeeze(string_to_hash(tf.cast(inputs[key], tf.string))) for key in sorted(inputs.keys())]
        input_concat = tf.stack(input_hashes, axis=-1)
        
        # Return both DenseFeatures output and hashed concatenated input info to enable verifying correct mapping
        return features_output, input_concat

def my_model_function():
    # Define example feature columns simulating 'condition', 'reviews' fields seen in the original issue
    # Assume both are categorical string columns with vocabulary lists (placeholders)
    # This mimics "f" in the dummy_model_2 example (feature_columns)
    
    # Note: vocabulary_list here is a made-up example for illustration
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="condition", vocabulary_list=["cold", "fever", "headache"]),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="reviews", vocabulary_list=["good", "bad", "average"])
    ]
    
    # Wrap categorical columns as indicator columns to convert to dense one-hot for DenseFeatures
    indicator_columns = [tf.feature_column.indicator_column(col) for col in feature_columns]
    
    model = MyModel(indicator_columns)
    
    # Compile model with dummy optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE')]
    )
    
    return model

def GetInput():
    # Return a dict of string tensors as input matching the keys expected by DenseFeatures layer
    # Input batch size B = 2 (arbitrary)
    inputs = {
        "condition": tf.constant(["cold", "fever"]),  # shape (2,), dtype string
        "reviews": tf.constant(["good", "bad"]),      # shape (2,), dtype string
    }
    return inputs

