# tf.random.uniform((5, ), dtype=tf.float32) for numeric inputs and tf.random.uniform((5, ), dtype=tf.string) for categorical input 'thal'
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the feature columns as in the original reproduction code.
        # Numeric features: "age" and "chol"
        self.numeric_columns = ["age", "chol"]
        # Categorical feature 'thal' with a dummy vocabulary list for demonstration
        # In a real scenario, the vocabulary would be the unique values of 'thal' from data.
        self.categorical_columns = {"thal": ["3", "6", "7"]}

        # Build feature columns from these
        self.feature_columns = {}

        # DenseFeatures layers for each feature column (to get separate feature outputs)
        self.dense_feature_layers = {}
        for feature_name in self.numeric_columns:
            # numeric_column shape=(), dtype float32
            col = tf.feature_column.numeric_column(feature_name)
            self.feature_columns[feature_name] = col
            self.dense_feature_layers[feature_name] = tf.keras.layers.DenseFeatures([col], name=f'{feature_name}_feature')

        for feature_name, vocab in self.categorical_columns.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
            ind_col = tf.feature_column.indicator_column(cat_col)
            self.feature_columns[feature_name] = ind_col
            self.dense_feature_layers[feature_name] = tf.keras.layers.DenseFeatures([ind_col], name=f'{feature_name}_feature')

        # Dense layers after concatenation of all features
        self.concat = tf.keras.layers.Concatenate()
        self.dense0 = tf.keras.layers.Dense(24, activation='relu', name='dense_0')
        self.dense1 = tf.keras.layers.Dense(24, activation='relu', name='dense_1')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax', name='output_layer')

    def call(self, inputs, training=False):
        # The inputs dict keys must match exactly the feature columns 
        # (numeric float32 for 'age', 'chol'; string for 'thal').
        # We extract features for each column separately with DenseFeatures layers
        feature_tensors = []
        for feature_name in sorted(self.feature_columns.keys()):
            # Each layer expects a dict input (in TF2 DenseFeatures works on dict inputs)
            # Passing full inputs dict is ok but we do explicit slicing so dtype/order is correct
            # Based on the issue, order and types matter, so we pass the full dict,
            # but inside DenseFeatures it picks by key name.
            x = self.dense_feature_layers[feature_name](inputs)
            feature_tensors.append(x)

        # Concatenate all feature slices to a single tensor
        x = self.concat(feature_tensors)
        x = self.dense0(x)
        x = self.dense1(x)
        y = self.output_layer(x)
        return y


def my_model_function():
    # Create and compile the model instance
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        run_eagerly=False  # compatible with tf.function jit_compile
    )
    return model


def GetInput():
    # Prepare a batch of inputs matching model's expected input dict
    # Assumptions based on issue:
    # batch_size = 5
    batch_size = 5
    # Numeric columns float32
    age = tf.random.uniform(shape=(batch_size,), minval=20, maxval=90, dtype=tf.float32)
    chol = tf.random.uniform(shape=(batch_size,), minval=100, maxval=300, dtype=tf.float32)

    # Categorical column 'thal' as strings, loading from a fixed vocabulary ["3", "6", "7"]
    # Pick randomly from vocab
    thal_vocab = ["3", "6", "7"]
    import numpy as np
    indices = np.random.choice(len(thal_vocab), size=batch_size)
    thal = tf.constant([thal_vocab[i] for i in indices])

    # Return dictionary input as expected by model.call()
    return {
        "age": age,
        "chol": chol,
        "thal": thal
    }

