# tf.random.uniform((B,), dtype={'age': tf.float32, 'trestbps': tf.float32, 'chol': tf.float32,
#                               'thalach': tf.float32, 'oldpeak': tf.float32, 'slope': tf.float32,
#                               'ca': tf.float32, 'thal': tf.string})

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Define feature columns matching the dataset columns
        self.age = feature_column.numeric_column("age")
        self.trestbps = feature_column.numeric_column("trestbps")
        self.chol = feature_column.numeric_column("chol")
        self.thalach = feature_column.numeric_column("thalach")
        self.oldpeak = feature_column.numeric_column("oldpeak")
        self.slope = feature_column.numeric_column("slope")
        self.ca = feature_column.numeric_column("ca")

        # Bucketized column for age
        self.age_buckets = feature_column.bucketized_column(
            self.age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

        # Categorical feature with vocabulary
        self.thal = feature_column.categorical_column_with_vocabulary_list(
            'thal', ['fixed', 'normal', 'reversible'])
        self.thal_one_hot = feature_column.indicator_column(self.thal)
        self.thal_embedding = feature_column.embedding_column(self.thal, dimension=8)

        # Crossed feature column (age buckets crossed with thal)
        crossed = feature_column.crossed_column([self.age_buckets, self.thal], hash_bucket_size=1000)
        crossed_ind = feature_column.indicator_column(crossed)

        # Combine all feature columns to DenseFeatures layer
        self.feature_columns = [
            self.age, self.trestbps, self.chol, self.thalach,
            self.oldpeak, self.slope, self.ca,
            self.age_buckets,
            self.thal_one_hot,
            self.thal_embedding,
            crossed_ind
        ]

        self.feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns)

        # Define the dense layers of the model
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs is expected to be a dict of feature tensors
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a dict input matching the expected input features of MyModel
    # For demonstration, batch size = 5, and generate dummy/random data with correct dtypes

    batch_size = 5

    import numpy as np

    # Numeric features: float32 values, shape (batch_size, 1)
    numeric_features = {
        "age": tf.random.uniform([batch_size, 1], 20, 70, dtype=tf.float32),
        "trestbps": tf.random.uniform([batch_size, 1], 90, 200, dtype=tf.float32),
        "chol": tf.random.uniform([batch_size, 1], 100, 400, dtype=tf.float32),
        "thalach": tf.random.uniform([batch_size, 1], 70, 210, dtype=tf.float32),
        "oldpeak": tf.random.uniform([batch_size, 1], 0, 5, dtype=tf.float32),
        "slope": tf.random.uniform([batch_size, 1], 0, 3, dtype=tf.float32),
        "ca": tf.random.uniform([batch_size, 1], 0, 4, dtype=tf.float32),
    }

    # Categorical 'thal' feature: shape (batch_size, 1), dtype string,
    # randomly pick from ['fixed', 'normal', 'reversible']
    thal_categories = ['fixed', 'normal', 'reversible']
    thal_values = np.random.choice(thal_categories, size=(batch_size, 1))
    thal_tensor = tf.convert_to_tensor(thal_values, dtype=tf.string)

    inputs = numeric_features
    inputs['thal'] = thal_tensor

    return inputs

