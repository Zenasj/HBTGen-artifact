# tf.Tensor with shape (?,) containing dict inputs with keys: 'age', 'n_siblings_spouses', 'parch', 'fare',
# 'sex', 'class', 'deck', 'embark_town', 'alone' (batch size is dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Define numeric feature keys
        self.numeric_features = ['age', 'n_siblings_spouses', 'parch', 'fare']

        # Define categorical features and their vocabularies
        self.categories = {
            'sex': ['male', 'female'],
            'class': ['First', 'Second', 'Third'],
            'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
            'alone': ['y', 'n']
        }

        # Create NumericColumns as feature columns
        self.numeric_columns = []
        for feature in self.numeric_features:
            self.numeric_columns.append(tf.feature_column.numeric_column(feature))

        # Create Categorical Indicator Columns
        self.categorical_columns = []
        for feature, vocab in self.categories.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
            self.categorical_columns.append(tf.feature_column.indicator_column(cat_col))

        # Layers to convert feature columns to tensors
        # Because tf.keras.layers.DenseFeatures expects a list of columns
        self.numeric_feature_layer = tf.keras.layers.DenseFeatures(self.numeric_columns)
        self.categorical_feature_layer = tf.keras.layers.DenseFeatures(self.categorical_columns)

        # Build the linear and DNN parts similar to DNNLinearCombinedClassifier
        # Linear model on the categorical features (represented as dense indicators)
        # DNN model on numeric features

        # Linear part is a single Dense layer with no activation
        self.linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            name='linear_layer')

        # DNN part uses two hidden layers (100 and 50 units) with ReLU as in original
        self.dnn_hidden1 = tf.keras.layers.Dense(100, activation='relu')
        self.dnn_hidden2 = tf.keras.layers.Dense(50, activation='relu')
        self.dnn_output = tf.keras.layers.Dense(1, activation=None)  # output logits

    def call(self, inputs):
        # inputs is a dictionary of features with shape (batch_size,)

        # Extract numeric features dictionary for numeric layer
        numeric_inputs = {key: tf.expand_dims(inputs[key], axis=-1)
                          if len(inputs[key].shape) == 1 else inputs[key] 
                          for key in self.numeric_features}

        # Extract categorical features dictionary for categorical layer
        categorical_inputs = {key: inputs[key] for key in self.categories.keys()}

        # Pass numeric features through numeric feature layer
        numeric_processed = self.numeric_feature_layer(numeric_inputs)  # shape (batch_size, num_numeric_features)

        # Pass categorical features through categorical feature layer
        categorical_processed = self.categorical_feature_layer(categorical_inputs)  # shape (batch_size, total_cat_indicators)

        # Linear logits on categorical indicators
        linear_logits = self.linear_layer(categorical_processed)  # shape (batch_size, 1)

        # DNN logits on numeric features
        x = self.dnn_hidden1(numeric_processed)
        x = self.dnn_hidden2(x)
        dnn_logits = self.dnn_output(x)  # shape (batch_size, 1)

        # Combine linear and dnn logits (sum)
        logits = linear_logits + dnn_logits  # shape (batch_size, 1)

        # Return logits as output (before sigmoid)
        return logits


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a batch of random input features matching the expected inputs to MyModel
    batch_size = 4  # arbitrary batch size for example

    import numpy as np

    # Numeric features: Float values, reasonable ranges inferred from Titanic data domain
    numeric_data = {
        'age': tf.random.uniform((batch_size,), minval=0, maxval=80, dtype=tf.float32),
        'n_siblings_spouses': tf.random.uniform((batch_size,), minval=0, maxval=8, dtype=tf.float32),
        'parch': tf.random.uniform((batch_size,), minval=0, maxval=6, dtype=tf.float32),
        'fare': tf.random.uniform((batch_size,), minval=0, maxval=500, dtype=tf.float32),
    }

    # Categorical features: random selection from vocabulary lists, encoded as string tensors
    categories = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }

    cat_data = {}
    for key, vocab in categories.items():
        # Random indices in vocab
        indices = tf.random.uniform((batch_size,), minval=0, maxval=len(vocab), dtype=tf.int32)
        # Gather vocab strings by indices
        tensor_values = tf.gather(vocab, indices)
        cat_data[key] = tf.convert_to_tensor(tensor_values)

    # Combine numeric and categorical features into one dictionary
    inputs = {}
    inputs.update(numeric_data)
    inputs.update(cat_data)

    return inputs

