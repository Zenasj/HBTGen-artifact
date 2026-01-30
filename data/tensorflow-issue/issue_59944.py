import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

num_outputs = 32 #hyperparameter
K= 32 #hyperparameter, size of embedding vector
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.2, ),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
  
])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.2, ),
    tf.keras.layers.Dense(num_outputs, activation='linear'), 
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(1,)) 
u_emb = Embedding(N, K)(input_user) # output is (num_samples, 1, K)
u_emb = Flatten()(u_emb)  # now it's (num_samples, K)

vu = user_NN(u_emb)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1) 

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()


history = model.fit([users, movies], ratings,callbacks=[callback], epochs=1000,batch_size=256,verbose=1,shuffle=True)


num_item_features=movies.shape[1]
users=df3['userId'].to_numpy()
ratings=df3['rating'].to_numpy()