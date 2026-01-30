import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_bayesian_wide_and_deep_model(wide_inputs, wide_feature_columns, 
                        deep_inputs, dnn_feature_columns, dnn_hidden_units, 
                        multihead_count = 64, p_value=0.5):

    #Build the Deep Network
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(deep_inputs)    

    for layerno, numnodes in enumerate(dnn_hidden_units):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno+1))(deep)
    
    deep = tf.keras.Model(inputs=deep_inputs, outputs=deep)
    
    #Build the Wide Network
    wide = tf.keras.layers.DenseFeatures(wide_feature_columns, name='wide_inputs')(wide_inputs)
    wide = tf.keras.layers.Dense(1, activation="linear", name="wide_output")(wide)
    wide = tf.keras.Model(inputs=wide_inputs, outputs=wide)

    #Concatenate the Wide & Deep
    both = tf.keras.layers.concatenate([deep.output, wide.output], name='both')

    #Create the multi-head layer
    multihead_pre_dropout = tf.keras.layers.Dropout(p_value)(both, training=True)
    multihead = tf.keras.layers.Dense(multihead_count, activation='relu', name='multihead')(multihead_pre_dropout)
    multihead_dropout = tf.keras.layers.Dropout(p_value)(multihead, training=True)

    #Create the output layer
    output = tf.keras.layers.Dense(2, activation='softmax', name='optimal_action')(multihead_dropout)
    
    #Convert the 2 inputs dictionary into a single list
    full_inputs = (wide.input).copy()
    full_inputs.update(deep.input)
    model = tf.keras.Model(inputs=full_inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

bwdmodel = build_bayesian_wide_and_deep_model(wide_inputs = crossed_columns_input, wide_feature_columns = crossed_columns,  deep_inputs = deep_columns_input, dnn_feature_columns = deep_columns, dnn_hidden_units = [512, 256, 128], multihead_count = 64, p_value=0.5)

#Build a list of crossed columns
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

#Wrap an indicator column around each of them to make them compatible with Keras
crossed_columns = [tf.feature_column.indicator_column(col) for col in crossed_columns_orig]

#Listing out dataframe column names to create Input layer tensors for wide part of network
col_names_for_crossed_columns = ["riid", 'campaign_category', "discount", "is_one_for_free",
                                 "free_shipping", "is_exclusive", "has_urgency", 'sl_contains_price',
                                 'is_discount_mentioned', 'sent_week', 'sent_dayofweek', 'sent_hr']

#Create feature dictionary to be passed for wide part of network
crossed_columns_input = {}
for col_name in col_names_for_crossed_columns:
  if col_name == 'campaign_category':
    crossed_columns_input[col_name] = tf.keras.Input(shape=(), name=col_name, dtype=tf.string)
  else:
    crossed_columns_input[col_name] = tf.keras.Input(shape=(), name=col_name, dtype=tf.int64)

"""Numeric Features"""
#Initialize list and dictionary to store feature columns and corresponding input tensors
numeric_feature_layer = []
numeric_feature_layer_input = {}

#Listing out df column names to create both Input layer tensors and numeric feature columns for deep part of network
numeric_feature_col_names = ['retention_score','recency_score','frequency_score',
                             'sent_week','sent_dayofweek','sent_hr','discount',
                             'sends_since_last_open']

#Build numeric feature columns and corresponding inputs, to be fed to deep part
for feature in numeric_feature_col_names:

  if feature in ['retention_score','recency_score','frequency_score']:
    numeric_feature_col = tf.feature_column.numeric_column(feature, dtype = tf.float32)
    numeric_feature_layer_input[feature] = tf.keras.Input(shape=(), name=feature, dtype=tf.float32)
  else:
    numeric_feature_col = tf.feature_column.numeric_column(feature, dtype = tf.int64)
    numeric_feature_layer_input[feature] = tf.keras.Input(shape=(), name=feature, dtype=tf.int64)

  numeric_feature_layer.append(numeric_feature_col)

"""Categorical Features (OHE)"""

#Initialize list and dictionary to store ohe categorical feature columns & corresponding input tensors
categorical_feature_layer = []
categorical_feature_layer_input = {}

#Listing out df column names to create both Input layer tensors and ohe categorical feature columns for deep part of network
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

#Build ohe categorical feature columns and corresponding inputs, to be fed to deep part
for (feature, vocab) in CATEGORIES.items():
  if feature == "campaign_category":  
    categorical_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab, dtype=tf.string)
    categorical_feature_layer_input[feature] = tf.keras.Input(shape=(), name=feature, dtype=tf.string)
  else:
    categorical_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab, dtype=tf.int64)
    categorical_feature_layer_input[feature] = tf.keras.Input(shape=(), name=feature, dtype=tf.int64)
  
  categorical_feature_col_ind = tf.feature_column.indicator_column(categorical_feature_col)
  categorical_feature_layer.append(categorical_feature_col_ind)

""""Categorical Features (Embedding)"""
#Initialize list and dictionary to store embedding categorical feature column & corresponding input tensors
embedding_feature_layer = []
embedding_feature_layer_input = {}

#Build embedding categorical feature column and corresponding input for riid column in df, to be fed to deep part
riid = tf.feature_column.categorical_column_with_hash_bucket("riid", hash_bucket_size=2000000, dtype=tf.int64)
riid_embedding = tf.feature_column.embedding_column(riid, dimension=32)
embedding_feature_layer.append(riid_embedding)
embedding_feature_layer_input["riid"] = tf.keras.Input(shape=(), name="riid", dtype=tf.int64)

"""Creating deep_columns and deep_columns_input"""
deep_columns = numeric_feature_layer + categorical_feature_layer + embedding_feature_layer
deep_columns_input = {**numeric_feature_layer_input, **categorical_feature_layer_input, **embedding_feature_layer_input}

#C1: Share the super set of inputs with both wide and part parts, instead of 2 separate wide and deep inputs
def build_bayesian_wide_and_deep_model(inputs, wide_feature_columns, dnn_feature_columns, dnn_hidden_units, multihead_count = 64, p_value=0.5):
    
    #C2: Do not build a tf.keras.Model(...) for the wide and deep parts separately for now
    #Building a tf.keras.Model(...) requires one to explicitly specify an additional tf.keras.Input(...) not just for the
    #W&D parts (which is anyways "inputs") but also for the layers that follow the W&D parts
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)    

    for layerno, numnodes in enumerate(dnn_hidden_units):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno+1))(deep)
    
    wide = tf.keras.layers.DenseFeatures(wide_feature_columns, name='wide_inputs')(inputs)

    both = tf.keras.layers.concatenate([wide, deep], name='both')

    pre_multihead_dropout = tf.keras.layers.Dropout(p_value)(both, training=True)
    multihead = tf.keras.layers.Dense(multihead_count, activation='relu', name='multihead')(pre_multihead_dropout)
    post_multihead_dropout = tf.keras.layers.Dropout(p_value)(multihead, training=True)

    output = tf.keras.layers.Dense(2, activation='softmax', name='optimal_action')(post_multihead_dropout)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model