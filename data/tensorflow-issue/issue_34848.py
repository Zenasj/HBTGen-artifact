# tf.random.uniform((B, )) ← Assumed input is a batch of dicts with features from heart.csv dataset (structured tabular data)

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, hyper_parameters):
        super().__init__()
        # Define feature columns as per define_features() logic
        self.feature_columns = []
        # Numeric columns from dataset
        for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
            self.feature_columns.append(feature_column.numeric_column(header))
        
        # Bucketized column for age
        age = feature_column.numeric_column("age")
        age_buckets = feature_column.bucketized_column(
            age,
            boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        )
        # age_buckets column is defined but not appended. Keep consistent with original code: commented out
        # self.feature_columns.append(age_buckets)

        # Categorical column with vocabulary for 'thal'
        thal = feature_column.categorical_column_with_vocabulary_list(
            'thal', ['fixed', 'normal', 'reversible']
        )
        
        # Embedding column for 'thal'
        thal_embedding = feature_column.embedding_column(
            thal, dimension=hyper_parameters['thal_embedding_cols']
        )
        # thal_embedding also commented out in original, keep consistent.
        # self.feature_columns.append(thal_embedding)

        # Crossed feature column (age_buckets X thal) with hash bucket and indicator column
        crossed_feature = feature_column.crossed_column(
            [age_buckets, thal], hash_bucket_size=1000
        )
        crossed_feature = feature_column.indicator_column(crossed_feature)
        # Commented out in original example
        # self.feature_columns.append(crossed_feature)

        # Create feature layer for the feature columns
        self.feature_layer = layers.DenseFeatures(self.feature_columns)
        
        # Build the sequential layers as in build_model()
        self.dense1 = layers.Dense(hyper_parameters['nodes'], activation='relu')
        self.dense2 = layers.Dense(hyper_parameters['nodes'], activation='relu')
        self.dense3 = layers.Dense(hyper_parameters['nodes'], activation='relu')
        self.dense4 = layers.Dense(hyper_parameters['nodes'], activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs: dictionary of feature_name: tensor (batch)
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = self.output_layer(x)
        return output


def my_model_function():
    # Return instance of MyModel with reasonable default hyperparameters
    hparams = {"nodes": 128, "thal_embedding_cols": 3}
    return MyModel(hparams)


def GetInput():
    """
    Return a random batched input dictionary simulating one batch of the input features 
    for the heart dataset used in the issue.
    
    Dataset features include:
    - age: numeric
    - trestbps: numeric
    - chol: numeric
    - thalach: numeric
    - oldpeak: numeric
    - slope: numeric
    - ca: numeric
    - thal: categorical with 3 classes: 'fixed', 'normal', 'reversible'
    
    The batch size is assumed as 32.
    
    Since tf.keras.layers.DenseFeatures expects dict of tensors keyed by feature names,
    we produce a dict of tf.Tensor of shape (batch_size, ) with proper dtypes.
    
    For categorical 'thal', it must be string tensor.
    
    """
    batch_size = 32

    # Random numeric features with plausible ranges

    import numpy as np

    # For reproducibility
    np.random.seed(0)

    age = tf.constant(np.random.randint(18, 70, size=(batch_size,)), dtype=tf.float32)
    trestbps = tf.constant(np.random.randint(90, 180, size=(batch_size,)), dtype=tf.float32)
    chol = tf.constant(np.random.randint(100, 400, size=(batch_size,)), dtype=tf.float32)
    thalach = tf.constant(np.random.randint(100, 200, size=(batch_size,)), dtype=tf.float32)
    oldpeak = tf.constant(np.random.uniform(0.0, 6.0, size=(batch_size,)), dtype=tf.float32)
    slope = tf.constant(np.random.randint(1, 3, size=(batch_size,)), dtype=tf.float32)
    ca = tf.constant(np.random.randint(0, 4, size=(batch_size,)), dtype=tf.float32)
    
    # For categorical 'thal', randomly pick from valid vocabulary
    thal_options = ['fixed', 'normal', 'reversible']
    thal_np = np.random.choice(thal_options, size=(batch_size,))
    thal = tf.constant(thal_np, dtype=tf.string)

    input_dict = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return input_dict

