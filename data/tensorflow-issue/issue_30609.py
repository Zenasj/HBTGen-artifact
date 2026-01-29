# tf.random.uniform((12,), dtype=tf.string) for categorical features, plus numerical features (floats), input shape is a dict of 8 keys, batch size 12

import functools
import tensorflow as tf

# Categories and means based on Titanic dataset example
CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n'],
}

MEANS = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399,
}

# We will assume input is a dictionary with keys matching CATEGORIES keys + MEANS keys (total 8 keys)
# Categorical features are string tensors of shape (batch_size,)
# Numerical features are float tensors of shape (batch_size,)

def process_continuous_data(mean, data):
    # Normalize continuous data by scaling with 1/(2*mean)
    data = tf.cast(data, tf.float32) * (1 / (2 * mean))
    return tf.reshape(data, [-1, 1])

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Build categorical feature columns with vocabularies
        categorical_columns = []
        for feature, vocab in CATEGORIES.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
            categorical_columns.append(tf.feature_column.indicator_column(cat_col))
        
        # Build numeric columns with normalizer_fn using process_continuous_data
        numerical_columns = []
        for feature in MEANS.keys():
            num_col = tf.feature_column.numeric_column(
                feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
            numerical_columns.append(num_col)
        
        # Combine categorical and numerical columns
        self.preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)
        
        # Build a simple sequential model after preprocessing
        # Using two hidden layers with 100 units and ReLU activations as in example
        self.hidden1 = tf.keras.layers.Dense(100, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(100, activation='relu')
        # Output layer with one unit (for binary classification), no activation specified here (logits)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None):
        # inputs is a dictionary of features matching the keys in CATEGORIES and MEANS
        x = self.preprocessing_layer(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly to original example
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of 12 input samples as a dictionary of tensors matching the feature columns
    
    batch_size = 12
    
    # For categorical columns, create string tensors with random choices from vocab lists
    inputs = {}
    for feature, vocab in CATEGORIES.items():
        # Pick random ints to index vocab, then map to string tensor
        indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(vocab), dtype=tf.int32)
        # Convert indices to the corresponding string from vocab
        # tf.gather does not work natively on Python list of strings, convert vocab to tf.constant
        vocab_const = tf.constant(vocab)
        inputs[feature] = tf.gather(vocab_const, indices)
    
    # For numerical features, create random floats near the means, use uniform around mean ± 50%
    for feature, mean in MEANS.items():
        low = mean * 0.5
        high = mean * 1.5
        inputs[feature] = tf.random.uniform(shape=(batch_size,), minval=low, maxval=high, dtype=tf.float32)
    
    return inputs

