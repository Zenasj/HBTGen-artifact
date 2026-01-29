# tf.random.uniform((B,), dtype=...) where B=batch size, input shape is dict of scalar features (string or int64 or float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Bayesian Wide and Deep model with MC Dropout.
    This unified model takes a *single* dict input containing all features used by both
    wide (crossed columns) and deep (numeric, categorical, embeddings) parts.
    It applies feature columns accordingly, runs the DNN and wide parts,
    then concatenates and applies dropout and multihead layers,
    finally outputs 2-class softmax.
    """

    def __init__(self, wide_feature_columns, dnn_feature_columns, dnn_hidden_units, 
                 multihead_count=64, p_value=0.5, **kwargs):
        super().__init__(**kwargs)
        self.wide_feature_columns = wide_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.dnn_hidden_units = dnn_hidden_units
        self.multihead_count = multihead_count
        self.p_value = p_value

        # DenseFeatures layers for wide and deep inputs
        self.wide_dense_features = tf.keras.layers.DenseFeatures(wide_feature_columns, name='wide_inputs')
        self.deep_dense_features = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')

        # DNN hidden layers
        self.dnn_layers = []
        for layerno, numnodes in enumerate(dnn_hidden_units):
            self.dnn_layers.append(
                tf.keras.layers.Dense(numnodes, activation='relu', name=f'dnn_{layerno + 1}')
            )

        # Multihead layers with MC Dropout
        self.dropout1 = tf.keras.layers.Dropout(p_value)
        self.multihead_dense = tf.keras.layers.Dense(multihead_count, activation='relu', name='multihead')
        self.dropout2 = tf.keras.layers.Dropout(p_value)

        # Final output layer
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax', name='optimal_action')

    def call(self, inputs, training=False):
        # Apply wide feature columns
        wide_out = self.wide_dense_features(inputs)

        # Apply deep feature columns
        deep_out = self.deep_dense_features(inputs)
        for layer in self.dnn_layers:
            deep_out = layer(deep_out)

        # Concatenate wide and deep parts
        both = tf.keras.layers.concatenate([wide_out, deep_out], name='both')

        # Apply MC Dropout layers (forced training=True to simulate Bayesian dropout)
        x = self.dropout1(both, training=True)
        x = self.multihead_dense(x)
        x = self.dropout2(x, training=True)

        # Output layer
        output = self.output_layer(x)
        return output


def my_model_function():
    """
    Build and return a MyModel instance with the wide and deep feature columns, inputs and units
    as given in the issue. This function prepares the feature columns needed.
    """

    # Wide feature crossed columns (Indicator columns around crossed_column)
    crossed_columns_orig = [
        tf.feature_column.crossed_column(["riid", 'campaign_category'], hash_bucket_size=20000000),
        tf.feature_column.crossed_column(["riid", 'discount'], hash_bucket_size=10000000),
        tf.feature_column.crossed_column(["riid", 'is_one_for_free'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'free_shipping'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'is_exclusive'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'has_urgency'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'sl_contains_price'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'is_discount_mentioned'], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(["riid", 'sent_week'], hash_bucket_size=10000000),
        tf.feature_column.crossed_column(["riid", 'sent_dayofweek'], hash_bucket_size=60000000),
        tf.feature_column.crossed_column(["riid", 'sent_hr'], hash_bucket_size=50000000),
    ]
    crossed_columns = [tf.feature_column.indicator_column(col) for col in crossed_columns_orig]

    # Deep feature numeric columns
    numeric_feature_col_names = ['retention_score','recency_score','frequency_score',
                                'sent_week','sent_dayofweek','sent_hr','discount',
                                'sends_since_last_open']
    numeric_feature_layer = []
    for feature in numeric_feature_col_names:
        if feature in ['retention_score','recency_score','frequency_score']:
            numeric_feature_col = tf.feature_column.numeric_column(feature, dtype=tf.float32)
        else:
            numeric_feature_col = tf.feature_column.numeric_column(feature, dtype=tf.int64)
        numeric_feature_layer.append(numeric_feature_col)

    # Deep feature categorical columns (one-hot encoding)
    CATEGORIES = {
        'promo' : [0, 1],
        'sale' : [0, 1],
        'campaign_category': ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10'],
        'is_one_for_free': [0, 1],
        'free_shipping': [0, 1],
        'is_exclusive': [0, 1],
        'has_urgency': [0, 1],
        'sl_contains_price': [0, 1],
        'is_discount_mentioned': [0, 1],
    }

    categorical_feature_layer = []
    for feature, vocab in CATEGORIES.items():
        if feature == "campaign_category":
            categorical_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab, dtype=tf.string)
        else:
            categorical_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab, dtype=tf.int64)
        categorical_feature_layer.append(tf.feature_column.indicator_column(categorical_feature_col))

    # Deep feature embedding columns
    riid = tf.feature_column.categorical_column_with_hash_bucket("riid", hash_bucket_size=2000000, dtype=tf.int64)
    riid_embedding = tf.feature_column.embedding_column(riid, dimension=32)

    deep_columns = numeric_feature_layer + categorical_feature_layer + [riid_embedding]

    # Return the model instance
    return MyModel(
        wide_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[512, 256, 128],
        multihead_count=64,
        p_value=0.5,
    )


def GetInput():
    """
    Prepare a dict of input tensors matching the inputs required by the model.
    This includes the full union of features needed for both wide and deep parts,
    with appropriate dtypes and scalar shapes.
    """
    # Wide feature columns input names & dtypes from crossed_columns_input keys and types in original
    # Strings and ints for wide inputs features
    input_dict = {}

    # 1) Wide input keys and types:
    col_names_for_crossed_columns = [
        "riid", 'campaign_category', "discount", "is_one_for_free",
        "free_shipping", "is_exclusive", "has_urgency", 'sl_contains_price',
        'is_discount_mentioned', 'sent_week', 'sent_dayofweek', 'sent_hr'
    ]

    for col_name in col_names_for_crossed_columns:
        if col_name == 'campaign_category':
            # string scalar input
            input_dict[col_name] = tf.random.uniform(shape=(), maxval=1, dtype=tf.string)  # placeholders for string, use tf.constant
            # Since tf.random.uniform does NOT support dtype=string, will create dummy constant below
            input_dict[col_name] = tf.constant(["CC1"])
        else:
            # int64 scalar input
            input_dict[col_name] = tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int64)

    # 2) Deep numeric inputs (some overlap with wide)
    numeric_feature_col_names = ['retention_score','recency_score','frequency_score',
                                'sent_week','sent_dayofweek','sent_hr','discount',
                                'sends_since_last_open']
    # Only add if not already in input_dict
    for feature in numeric_feature_col_names:
        if feature in input_dict:
            # Already assigned above as int64 - maybe need to assign float32 for some
            if feature in ['retention_score','recency_score','frequency_score']:
                input_dict[feature] = tf.random.uniform(shape=(), dtype=tf.float32)
            else:
                # keep existing int64
                pass
        else:
            if feature in ['retention_score','recency_score','frequency_score']:
                input_dict[feature] = tf.random.uniform(shape=(), dtype=tf.float32)
            else:
                input_dict[feature] = tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int64)

    # 3) Deep categorical inputs (one-hot categories)
    CATEGORIES = {
        'promo' : [0, 1],
        'sale' : [0, 1],
        'campaign_category': ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'CC7', 'CC8', 'CC9', 'CC10'],
        'is_one_for_free': [0, 1],
        'free_shipping': [0, 1],
        'is_exclusive': [0, 1],
        'has_urgency': [0, 1],
        'sl_contains_price': [0, 1],
        'is_discount_mentioned': [0, 1],
    }
    for feature, vocab in CATEGORIES.items():
        if feature == "campaign_category":
            # Override string constant used above with consistent value
            input_dict[feature] = tf.constant(["CC1"])
        else:
            input_dict[feature] = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int64)

    # 4) Embedding feature: "riid" int64 hash bucket
    input_dict["riid"] = tf.random.uniform(shape=(), minval=0, maxval=100000, dtype=tf.int64)

    return input_dict

