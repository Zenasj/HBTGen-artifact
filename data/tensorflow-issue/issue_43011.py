from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
inputs = {'feature_alpha' : tf.keras.layers.Input(name='feature_alpha', 
                                                 shape=(None,), 
                                                 sparse=True, 
                                                 dtype=tf.dtypes.string),
         'feature_beta' : tf.keras.layers.Input(name='feature_beta', 
                                                 shape=(None,), 
                                                 sparse=True, 
                                                 dtype=tf.dtypes.string)}
def gen_model(inputs):
    feature_alpha = tf.feature_column.categorical_column_with_hash_bucket('feature_alpha', 100, dtype=tf.dtypes.string)
    feature_beta = tf.feature_column.categorical_column_with_hash_bucket('feature_beta', 200, dtype=tf.dtypes.string)    
    
    alpha_emb = tf.feature_column.embedding_column(feature_alpha, dimension=10)
    beta_emb = tf.feature_column.embedding_column(feature_beta, dimension=20)
    out = tf.keras.layers.DenseFeatures([alpha_emb, beta_emb])(inputs)

    out = tf.keras.layers.Dense(64, activation='relu')(out)

    model = tf.keras.Model(inputs, out)

    return model        

model = gen_model(inputs)

print(model.trainable_variables)
model.save('mdl.h5')