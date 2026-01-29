# tf.random.uniform((B,)) with features 'age_buckets' and 'thal' as categorical inputs, 
# eventually used in crossed_column and indicator_column (inputs represented as dict of tensors)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We define two categorical features: age_buckets and thal with assumed vocab sizes
        # These correspond to integer indices representing bucketized age and thalassemia status.

        # Assumptions about vocab sizes based on typical example datasets:
        age_bucket_vocab_size = 10  # e.g., 10 age buckets
        thal_vocab_size = 4         # e.g., 3 classes + out of vocab

        # Create categorical feature columns
        self.age_buckets = tf.feature_column.categorical_column_with_identity("age_buckets", num_buckets=age_bucket_vocab_size)
        self.thal = tf.feature_column.categorical_column_with_identity("thal", num_buckets=thal_vocab_size)
        
        # Cross the two categorical columns
        crossed = tf.feature_column.crossed_column([self.age_buckets, self.thal], hash_bucket_size=1000)
        crossed_indicator = tf.feature_column.indicator_column(crossed)

        # Create DenseFeatures layer to handle feature columns input dictionary
        self.feature_layer = tf.keras.layers.DenseFeatures([crossed_indicator])

        # Simple dense layers after features
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, features, training=False):
        """
        Args:
            features: dict of tensors, keys are 'age_buckets' and 'thal'.
                      Each tensor is expected to be integer indices of shape (batch_size,)
        Returns:
            A tensor of shape (batch_size, 1) with sigmoid activation
        """
        x = self.feature_layer(features)
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.output_layer(x)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    """
    Returns:
        A dictionary of tensors suitable as input to MyModel.call()
    Assumptions:
        batch size = 16 (arbitrary choice for example)
        age_buckets values uniformly sampled from [0, 9]
        thal values uniformly sampled from [0, 3]
    """
    batch_size = 16
    age_buckets = tf.random.uniform(shape=(batch_size,), minval=0, maxval=10, dtype=tf.int32)
    thal = tf.random.uniform(shape=(batch_size,), minval=0, maxval=4, dtype=tf.int32)
    return {"age_buckets": age_buckets, "thal": thal}

