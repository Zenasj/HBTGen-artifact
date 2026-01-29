# tf.random.uniform((B, None), dtype=tf.string) ← Input is batch of CSV strings (1D string tensor batch)
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import feature_column as tfc

# Hyperparameters and constants inferred/converted from original flags
TRAIN_SIZE = 100000  # Orig train size param (the actual dataset is smaller)
NUM_EPOCHS = 70
BATCH_SIZE = 5
NUM_EVAL = 20

LEARNING_DECAY_RATE = 0.7
HIDDEN_UNITS = [128, 64, 32, 16]
LEARNING_RATE = 0.00135
L1_REGULARIZATION = 0.0216647
L2_REGULARIZATION = 0.0673949
DROPOUT = 0.899732
SHUFFLE_BUFFER_SIZE = 10000

checkpoint_steps = int((TRAIN_SIZE / BATCH_SIZE) * (NUM_EPOCHS / NUM_EVAL))


class CLVFeatures(object):
    """
    Feature definitions and helper for numeric/categorical columns,
    mimicking the Tutorial's context.py CLVFeatures.
    """
    HEADERS = ['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
               'frequency_btyd', 'recency', 'T', 'time_between',
               'avg_basket_value', 'avg_basket_size', 'cnt_returns',
               'has_returned', 'frequency_btyd_clipped', 'monetary_btyd_clipped',
               'target_monetary_clipped', 'target_monetary']
    HEADERS_DEFAULT = [[''], [0.0], [0.0], [0],
                       [0], [0], [0], [0.0],
                       [0.0], [0.0], [0],
                       [-1], [0], [0.0],
                       [0.0], [0.0]]

    NUMERICS = {
        'monetary_dnn': [],
        'recency': [],
        'frequency_dnn': [],
        'T': [],
        'time_between': [],
        'avg_basket_value': [],
        'avg_basket_size': [],
        'cnt_returns': []
    }
    CATEGORICALS_W_LIST = {
        'has_returned': [0, 1]
    }
    CROSSED = []
    KEY = 'customer_id'
    UNUSED = [KEY,
              'monetary_btyd', 'frequency_btyd', 'frequency_btyd_clipped',
              'monetary_btyd_clipped', 'target_monetary_clipped']
    TARGET_NAME = 'target_monetary'

    def __init__(self, ignore_crosses=False):
        self.ignore_crosses = ignore_crosses
        self.headers, self.numerics_names, self.categorical_names = self._keep_used()
        self.continuous, self.categorical = self._make_base_features()
        if not self.ignore_crosses:
            self.crossed_for_wide, self.crossed_for_deep = self._make_crossed()

    def _keep_used(self):
        headers = [h for h in self.HEADERS if h not in self.UNUSED]
        numerics_names = {
            k: v for k, v in self.NUMERICS.items()
            if (k not in self.UNUSED) and (k != self.TARGET_NAME)
        }
        categorical_names = {
            k: v for k, v in self.CATEGORICALS_W_LIST.items()
            if k not in self.UNUSED
        }
        return headers, numerics_names, categorical_names

    def get_key(self):
        return self.KEY

    def get_used_headers(self, with_key=False, with_target=False):
        used_headers = [h for h in self.headers if h != self.TARGET_NAME]
        if with_key:
            used_headers.insert(0, self.KEY)
        if with_target:
            used_headers.append(self.TARGET_NAME)
        return used_headers

    def get_defaults(self, headers_names=None, with_key=False):
        if headers_names is None:
            headers_names = self.get_used_headers(with_key)
        keep_indexes = [self.HEADERS.index(n) for n in headers_names]
        return [self.HEADERS_DEFAULT[i] for i in keep_indexes]

    def get_all_names(self):
        return self.HEADERS

    def get_all_defaults(self):
        return self.HEADERS_DEFAULT

    def get_unused(self):
        return self.UNUSED

    def get_target_name(self):
        return self.TARGET_NAME

    def _make_base_features(self):
        continuous = {
            key_name: tfc.numeric_column(key_name)
            for key_name in self.numerics_names.keys()
        }
        categorical = {
            key_name: tfc.categorical_column_with_vocabulary_list(
                key=key_name,
                vocabulary_list=voc)
            for key_name, voc in self.categorical_names.items()
        }
        return continuous, categorical

    def get_base_features(self):
        # Fix typo from original code: should be self.continuous not continous
        return self.continuous, self.categorical

    def _prepare_for_crossing(self, key_name, num_bck, boundaries):
        key = None
        if key_name in self.continuous.keys():
            if boundaries is not None:
                key = tfc.bucketized_column(self.continuous[key_name], boundaries)
            else:
                key = tfc.categorical_column_with_identity(key_name, num_bck)
        elif key_name in self.categorical.keys():
            key = key_name
        else:
            key = key_name
        return key

    def _make_crossed(self):
        f_crossed_for_wide = []
        f_crossed_for_deep = []
        # The original CROSSED is empty, but logic is preserved if used
        for to_cross in self.CROSSED:
            keys = []
            bck_size = 1
            for (key, bck, bnd) in to_cross:
                keys.append(self._prepare_for_crossing(key, bck, bnd))
                bck_size *= bck

            t_crossed = tfc.crossed_column(keys, min(bck_size, 10000))
            t_dimension = int(bck_size ** 0.25)
            f_crossed_for_wide.append(t_crossed)
            f_crossed_for_deep.append(tfc.embedding_column(t_crossed, t_dimension))
        return f_crossed_for_wide, f_crossed_for_deep

    def get_wide_features(self):
        wide_features = list(self.categorical.values())
        if not self.ignore_crosses:
            wide_features += self.crossed_for_wide
        return wide_features

    def get_deep_features(self, with_continuous=True):
        deep_features = [tfc.indicator_column(f) for f in self.categorical.values()]
        if with_continuous:
            deep_features += list(self.continuous.values())
        if not self.ignore_crosses:
            deep_features += self.crossed_for_deep
        return deep_features


clvf = CLVFeatures(ignore_crosses=True)


class MyModel(tf.keras.Model):
    """
    Approximate the tutorial's DNNRegressor model as a Keras model.
    - Input: batch of dict features matching deep_features columns
    - Output: single scalar per example (regression output for target monetary)
    """

    def __init__(self):
        super().__init__()
        # Build input layers and embedding/indicator layers matching feature columns

        # We will build a keras input layer dict for each feature column name (for eager TF 2 style)

        # Note: feature columns are dense numeric or indicator categorical
        # So for continuous: numeric_column -> float inputs
        # For categorical: indicator_column -> one-hot encoded of category vocabulary size
        self.continuous_names = list(clvf.continuous.keys())
        self.categorical_names = list(clvf.categorical.keys())

        # To mimic feature_column.indicator_column outputs, we must embed categorical as one-hot vectors or embeddings.
        # For simplicity, indicator_column can be replaced by one-hot encoding via tf.one_hot

        # For construction, we use embedding layers or one-hot:

        # Mapping from categorical feature to vocab size:
        self.cat_vocab_sizes = {
            k: len(voc) for k, voc in clvf.categorical_names.items()
        }

        # Now build embedding layers for categorical features or one-hot; original used indicator_column so 
        # output shape is vocab size (one-hot vector).

        # For continuous numeric - input float vector as is.

        # Input layers - for serving batch inputs dict:
        self.inputs_dict = {}

        # Build embedding layers for categorical features (for indicator_column, we can simply one-hot encode)
        # Use one-hot encoding + dense projection layer as a simple equivalent
        self.cat_embed_proj_layers = {
            k: tf.keras.layers.Dense(len(voc), activation='linear', use_bias=False, name=f"onehot_proj_{k}")
            for k, voc in clvf.categorical_names.items()
        }

        # Create hidden dense layers per HIDDEN_UNITS
        self.hidden_layers = []
        for i, units in enumerate(HIDDEN_UNITS):
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f"dense_{i}"))
            self.hidden_layers.append(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
            self.hidden_layers.append(tf.keras.layers.Dropout(DROPOUT, name=f"dropout_{i}"))

        self.output_layer = tf.keras.layers.Dense(1, activation='linear', name="output")

    def call(self, inputs, training=False):
        """
        Inputs:
          inputs: dictionary mapping feature name to tensor batch of input values
        """
        # Process continuous inputs concatenation
        cont_inputs = []
        for name in self.continuous_names:
            # inputs[name] expected as [batch_size, 1] float tensor
            cont_inputs.append(tf.cast(tf.expand_dims(inputs[name], -1), tf.float32))
        if cont_inputs:
            cont_concat = tf.keras.layers.Concatenate(axis=-1)(cont_inputs)
        else:
            cont_concat = None

        # Process categorical inputs via one-hot projection layers
        cat_outputs = []
        for name in self.categorical_names:
            cat_tensor = inputs[name]  # expected integer tensor shape [batch_size]
            # one-hot encode the integer category index (assumed integer scalar per example)
            one_hot = tf.one_hot(cat_tensor, depth=self.cat_vocab_sizes[name], dtype=tf.float32)
            # project one-hot vector linearly via dense layer (like indicator_column)
            emb = self.cat_embed_proj_layers[name](one_hot)
            cat_outputs.append(emb)

        if cat_outputs:
            cat_concat = tf.keras.layers.Concatenate(axis=-1)(cat_outputs)
        else:
            cat_concat = None

        # Combine continuous and categorical features
        if cont_concat is not None and cat_concat is not None:
            x = tf.keras.layers.Concatenate(axis=-1)([cont_concat, cat_concat])
        elif cont_concat is not None:
            x = cont_concat
        elif cat_concat is not None:
            x = cat_concat
        else:
            raise ValueError("No input features provided")

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        # Output layer: regression prediction (batch, 1)
        output = self.output_layer(x)
        return tf.squeeze(output, axis=-1)  # shape (batch,)


def my_model_function():
    return MyModel()


def parse_csv(csv_row):
    """
    Parse CSV row string to feature dict tensor and target tensor.
    This mimics the original parse_csv in the Issue with updated API.
    """
    columns = tf.io.decode_csv(csv_row, record_defaults=clvf.get_all_defaults())
    features = dict(zip(clvf.get_all_names(), columns))
    for col in clvf.get_unused():
        features.pop(col)
    target = features.pop(clvf.get_target_name())
    return features, target


def preprocess_features(features):
    """
    Convert raw feature tensors extracted from CSV to correct dtype for MyModel inputs:
    - Continuous features: float32 (shape [batch])
    - Categorical features: int32 (shape [batch])
    """
    # Continuous numeric features to float32 scalar tensors
    cont_features = {}
    for name in clvf.continuous.keys():
        val = features.get(name)
        if val is not None:
            cont_features[name] = tf.cast(val, tf.float32)
    # Categorical features to int32 scalar tensors
    cat_features = {}
    for name in clvf.categorical.keys():
        val = features.get(name)
        if val is not None:
            # Original vocab was list of ints for categorical
            cat_features[name] = tf.cast(val, tf.int32)
    # Merge dicts for model input
    model_input = {**cont_features, **cat_features}
    return model_input


def GetInput():
    """
    Returns:
      A batch of CSV rows (strings) suitable as input for MyModel.
      Then parse these CSV rows, preprocess, and return the processed dict
      suitable as input to MyModel.
    """
    # Assumptions:
    # - We create a batch of 5 CSV rows with dummy values matching defaults.
    batch_size = BATCH_SIZE

    # Build CSV strings with header dropped (to mimic data lines)
    headers = clvf.get_all_names()
    defaults = clvf.get_all_defaults()

    # Use default values to compose CSV rows (as strings)
    # Convert defaults to string values for CSV cells
    def csv_str_from_default(def_val):
        if isinstance(def_val, list) or isinstance(def_val, tuple):
            val = def_val[0]
        else:
            val = def_val
        return str(val)

    # Compose one CSV row string from defaults
    csv_row = ','.join(csv_str_from_default(d) for d in defaults)

    # Create batch of identical CSV rows for input
    batch_csv_rows = tf.constant([csv_row] * batch_size)

    # Parse CSV rows -> features dict and target tensor
    features, targets = tf.map_fn(parse_csv,
                                 batch_csv_rows,
                                 dtype=(
                                     {name: tf.float32 if name in clvf.continuous else tf.int32
                                      for name in clvf.get_used_headers(with_key=True)},
                                     tf.float32
                                 ))

    # parse_csv returns features without target; make sure to cast properly
    # Alternative: simpler approach for tf 2:
    features_list = []
    target_list = []
    for row in batch_csv_rows:
        f, t = parse_csv(row)
        features_list.append(f)
        target_list.append(t)
    # Stack features per key
    model_inputs = {}
    for key in features_list[0].keys():
        vals = [f[key] for f in features_list]
        model_inputs[key] = tf.stack(vals)
    targets = tf.stack(target_list)

    # Preprocess features to right dtypes
    model_inputs = preprocess_features(model_inputs)

    return model_inputs


# Ensure the model, input, and forward pass works with XLA jit_compile:
# Example usage (not included per instructions):
# model = my_model_function()
# @tf.function(jit_compile=True)
# def compiled(x): return model(x)
#
# inp = GetInput()
# out = compiled(inp)

