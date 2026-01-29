# tf.random.uniform((batch_size,)) for numeric cols and tf.string tensors for categorical cols in input dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
from tensorflow import keras as k

class NUM_TO_DENSE(layers.Layer):
    def __init__(self, num_cols):
        super().__init__()
        self.keys = num_cols
        self.keys_all = self.keys + [str(i) + '__nullcol' for i in self.keys]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'keys': self.keys,
            'keys_all': self.keys_all,
        })
        return config

    def build(self, input_shape):
        def create_moving_mean_vars():
            return tf.Variable(initial_value=0., shape=(), dtype=tf.float32, trainable=False)
        self.moving_means_total = {t: create_moving_mean_vars() for t in self.keys}
        self.layer_global_counter = tf.Variable(initial_value=0., shape=(), dtype=tf.float32, trainable=False)

    def call(self, inputs, training=True):
        # inputs: dict of numeric columns keyed by col names, each is a tensor of shape (batch,)
        null_cols = {k: tf.math.is_finite(inputs[k]) for k in self.keys}
        current_means = {}

        def compute_update_current_means(t):
            current_mean = tf.math.divide_no_nan(
                tf.reduce_sum(tf.where(null_cols[t], inputs[t], 0.), axis=0),
                tf.reduce_sum(tf.cast(tf.math.is_finite(inputs[t]), tf.float32), axis=0)
            )
            self.moving_means_total[t].assign_add(current_mean)
            return current_mean

        if training:
            current_means = {t: compute_update_current_means(t) for t in self.keys}
            outputs = {t: tf.where(null_cols[t], inputs[t], current_means[t]) for t in self.keys}
            outputs.update({str(k) + '__nullcol': tf.cast(null_cols[k], tf.float32) for k in self.keys})
            self.layer_global_counter.assign_add(1.)
        else:
            outputs = {
                t: tf.where(
                    null_cols[t],
                    inputs[t],
                    (self.moving_means_total[t] / self.layer_global_counter)
                ) for t in self.keys
            }
            outputs.update({str(k) + '__nullcol': tf.cast(null_cols[k], tf.float32) for k in self.keys})

        # outputs: dict of tensors keyed by num cols + their null indicator cols
        return outputs

class PREPROCESS_MONSOON(layers.Layer):
    def __init__(self, cat_cols_with_unique_values, num_cols):
        '''
        cat_cols_with_unique_values: dict like {'col_cat': [unique_values_list]}
        num_cols: list of numerical column names
        '''
        super().__init__()
        self.cat_cols = cat_cols_with_unique_values
        self.num_cols = num_cols

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'cat_cols': self.cat_cols,
            'num_cols': self.num_cols,
        })
        return config

    def build(self, input_shape):
        self.ntd = NUM_TO_DENSE(self.num_cols)
        self.num_colnames = self.ntd.keys_all  # includes null cols
        # Create embedding columns for each categorical column
        self.ctd = {
            k: layers.DenseFeatures(
                feature_column.embedding_column(
                    feature_column.categorical_column_with_vocabulary_list(k, v),
                    tf.cast(
                        tf.math.ceil(tf.math.log(tf.cast(len(v), tf.float32))),
                        tf.int32
                    ).numpy()
                )
            ) for k, v in self.cat_cols.items()
        }
        self.cat_colnames = list(self.cat_cols.keys())
        self.dense_colnames = self.num_colnames + self.cat_colnames

    def call(self, inputs, training=True):
        # inputs: dict of tensors keyed by all feature column names
        # Numeric preprocessing
        dense_num_d = self.ntd(inputs, training=training)  # dict of tensors keyed by numeric keys + nullcols
        
        # Categorical embeddings
        dense_cat_d = {k: self.ctd[k](inputs) for k in self.cat_colnames}
        
        # Stack numeric cols (including null indicators) on axis=1
        dense_num = tf.stack([dense_num_d[k] for k in self.num_colnames], axis=1)
        
        # Concatenate categorical embeddings on axis=1
        dense_cat = tf.concat([dense_cat_d[k] for k in self.cat_colnames], axis=1)
        
        # Concatenate all dense vectors into one tensor (batch_size, total_features)
        dense_all = tf.concat([dense_num, dense_cat], axis=1)
        return dense_all


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Setup data types for example inputs inferred from dataset columns:
        # In this example, for demonstration, we assume:
        # - num_cols: a known list of strings (e.g. the 28*28 flattened MNIST numeric features)
        # - cat_cols_with_unique_values: dict with keys as categorical columns and their unique values list

        # Since the original dataset reading and CSV preparation is not reproducible easily,
        # We'll simulate with placeholders for num_cols and cat_cols_with_unique_values for input construction.

        # For demonstration, assume 3 numeric columns (arbitrary)
        self.num_cols = [f"numcol{i}" for i in range(3)]

        # Assume 2 categorical columns with vocabularies (simplified)
        self.cat_cols_with_unique_values = {
            "catcol1": ['a', 'b', 'c'],
            "catcol2": ['x', 'y', 'z', 'w']
        }

        # Instantiate the preprocessing layer
        self.preprocess = PREPROCESS_MONSOON(
            cat_cols_with_unique_values=self.cat_cols_with_unique_values,
            num_cols=self.num_cols
        )
        # BatchNormalization layer
        self.bn = layers.BatchNormalization()
        # Hidden dense layer
        self.dense1 = layers.Dense(10, activation='relu', name='dense_1')
        # Output layer with 10 classes (MNIST style), no activation (logits)
        self.predictions = layers.Dense(10, activation=None, name='output')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.preprocess(inputs, training=training)
        x = self.bn(x, training=training)
        x = self.dense1(x)
        x = self.predictions(x)
        return x

def my_model_function():
    # Return an instance of MyModel. 
    # Since the original code loads dataset and constructs from data, here we use placeholders.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The model expects a dict with keys equal to num_cols (float tensors) and cat_cols (string tensors)

    batch_size = 4  # arbitrary batch size for test

    # Numeric columns: random floats, shape (batch_size,)
    num_inputs = {
        k: tf.random.uniform((batch_size,), minval=0.0, maxval=1.0, dtype=tf.float32)
        for k in [f"numcol{i}" for i in range(3)]
    }

    # Categorical columns: random choice from vocabulary, shape (batch_size,)
    cat_cols_with_unique_values = {
        "catcol1": ['a', 'b', 'c'],
        "catcol2": ['x', 'y', 'z', 'w']
    }
    cat_inputs = {}
    for k, vocab in cat_cols_with_unique_values.items():
        # Tensor of strings with shape (batch_size,)
        values = tf.constant(vocab)
        rand_indices = tf.random.uniform((batch_size,), minval=0, maxval=len(vocab), dtype=tf.int32)
        cat_inputs[k] = tf.gather(values, rand_indices)

    # Combine all inputs into a single dict
    inputs = {**num_inputs, **cat_inputs}
    return inputs

