# tf.random.uniform((B,)) with each element a dictionary of scalar features (strings or floats/ints)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define feature columns 
        feature_columns = []

        # Numeric columns
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']
        for feature_name in numeric_features:
            feature_columns.append(tf.feature_column.numeric_column(feature_name))
        
        # Bucketized column for age
        age = tf.feature_column.numeric_column("age")
        age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        feature_columns.append(age_buckets)
        
        # Categorical column with vocabulary and indicator (one-hot)
        thal = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
        thal_one_hot = tf.feature_column.indicator_column(thal)
        feature_columns.append(thal_one_hot)
        
        # Embedding column on thal
        thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
        feature_columns.append(thal_embedding)
        
        # Crossed column example (age bucket crossed with thal), indicator column
        crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
        crossed_feature = tf.feature_column.indicator_column(crossed_feature)
        feature_columns.append(crossed_feature)

        # DenseFeatures layer to transform inputs dict to dense tensor
        self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        # Define dense network layers
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.final = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        inputs: dict of feature_name -> tensor of shape (batch_size, 1),
        where categorical string features are dtype string, and numeric are float/int.
        """
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.final(x)
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Construct a batch of example inputs that match expected feature signatures.
    # For simplicity, batch_size=4 here.

    batch_size = 4
    import numpy as np

    # Input dict keys must match feature columns:
    # Numeric features are floats shaped (batch_size, 1)
    numeric_features = {
        'age': np.random.uniform(20, 60, size=(batch_size, 1)).astype(np.float32),
        'trestbps': np.random.uniform(90, 140, size=(batch_size, 1)).astype(np.float32),
        'chol': np.random.uniform(100, 300, size=(batch_size, 1)).astype(np.float32),
        'thalach': np.random.uniform(100, 200, size=(batch_size, 1)).astype(np.float32),
        'oldpeak': np.random.uniform(0, 5, size=(batch_size, 1)).astype(np.float32),
        'slope': np.random.uniform(0, 3, size=(batch_size, 1)).astype(np.float32),
        'ca': np.random.uniform(0, 4, size=(batch_size, 1)).astype(np.float32)
    }

    # Categorical feature 'thal' must be strings shaped (batch_size, 1)
    # Choose randomly among ['fixed', 'normal', 'reversible']
    thal_options = np.array(['fixed', 'normal', 'reversible'])
    thal_idx = np.random.randint(0, 3, size=(batch_size,))
    thal_data = thal_options[thal_idx].reshape(batch_size, 1)

    inputs = {}
    inputs.update(numeric_features)
    inputs['thal'] = tf.constant(thal_data)

    # Wrap numpy arrays as tensors
    for k, v in inputs.items():
        if not isinstance(v, tf.Tensor):
            inputs[k] = tf.convert_to_tensor(v)

    return inputs

