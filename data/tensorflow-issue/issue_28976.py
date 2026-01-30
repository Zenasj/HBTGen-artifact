import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    store_feature = tf.feature_column.categorical_column_with_vocabulary_list('store', vocabulary_list=['a', 'b'])

    store_feature = tf.feature_column.embedding_column(store_feature, dimension=64)

    loc_feature = tf.feature_column.categorical_column_with_vocabulary_list('loc', vocabulary_list=['x', 'y', 'z'])

    loc_feature = tf.feature_column.embedding_column(loc_feature, dimension=32)

    inp_1 = tf.keras.Input(name='store', dtype=tf.string, shape=(1,))
    inp_2 = tf.keras.Input(name='loc', dtype=tf.string, shape=(1,))
    keras_dict_input = {'store': inp_1, 'loc': inp_2}
    x = tf.keras.layers.DenseFeatures(feature_columns=[store_feature, loc_feature])(keras_dict_input)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    model = tf.keras.Model(keras_dict_input, x)

    return model