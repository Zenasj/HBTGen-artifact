import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

Python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Flatten

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, alpha=0.5):
  assert len(layers) == len(reg_layers)
  num_layer = len(layers) #Number of layers in the MLP
  
  # Input variables
  user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
  item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
  
  # Embedding layer
  MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user', 
                                embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), 
                                input_length=1)
  MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item', 
                                embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), 
                                input_length=1)

  MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user", 
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), 
                                  input_length=1)
  MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item', 
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), 
                                  input_length=1)

  # MF part
  mf_user_latent = Flatten()(MF_Embedding_User(user_input))
  mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
  mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

  # MLP part
  mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
  mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
  mlp_vector = keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])

  for idx in range(1, num_layer):
    mlp_vector = Dense(layers[idx], 
                      activation='relu', 
                      kernel_regularizer = l2(reg_layers[idx]), 
                      bias_regularizer = l2(reg_layers[idx]), 
                      name="layer%d" %idx)(mlp_vector)

  # Concatenate MF and MLP parts
  mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
  mlp_vector = Lambda(lambda x : x * (1 - alpha))(mlp_vector)
  predict_vector = keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])

  # Final prediction layer
  prediction = Dense(1, 
                    activation='sigmoid', 
                    kernel_initializer='lecun_uniform', 
                    bias_initializer ='lecun_uniform', 
                    name = "prediction")(predict_vector)

  model = keras.Model(inputs=[user_input, item_input], outputs=[prediction])
  return model

def generate_data(num_user, num_item, count=100):
    user_input = []
    item_input = []
    labels = []
    for _ in range(count):
        user = np.random.randint(0,num_user)
        item = np.random.randint(0,num_item)
        label = np.random.randint(0,2)
        user_input.append(user)
        item_input.append(item)
        labels.append(label)
    return np.asarray(user_input), np.asarray(item_input), np.asarray(labels)

def test_model():
    num_user = 1000000
    num_item = 100000
    count = 10000
    user_input, item_input, labels = generate_data(num_user, num_item, count)

    model = get_model(num_user, num_item)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy()
    )

    # Callbacks
    callbacks = [ tf.keras.callbacks.TensorBoard(log_dir='tb-logs') ]
    model.fit([user_input, item_input], labels, batch_size=256, epochs=3, callbacks=callbacks)

if __name__ == "__main__":
    print("Tensorflow version: ", tf.__version__)
    test_model()

tf.__version__
'2.0.0-rc1'