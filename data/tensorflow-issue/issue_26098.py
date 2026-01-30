import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

my_loss = -tf.keras.backend.mean(critic_output)
with my_loss.graph.as_default():
    actor_updates = self.optimizer_actor.get_updates(params=self.actor.trainable_weights, loss=my_loss)
self.actor_train_on_batch = tf.keras.backend.function(inputs=[state_input], outputs=[self.actor(state_input)], updates=actor_updates)

my_loss = -tf.keras.backend.mean(critic_output)
actor_updates = self.optimizer_actor.get_updates(params=self.actor.trainable_weights, loss=my_loss)
self.actor_train_on_batch = tf.keras.backend.function(inputs=[state_input], outputs=[self.actor(state_input)], updates=actor_updates)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

my_model = Sequential([Dense(1, activation='relu', input_shape=(5,))])
my_loss = -tf.reduce_mean(my_model.output)
my_optim = Adam()

# with my_loss.graph.as_default(): # <-- THIS IS THE WORKAROUND
updates = my_optim.get_updates(params=my_model.trainable_weights, loss=my_loss)

my_fn = tf.keras.backend.function(inputs=my_model.inputs, outputs=my_model.outputs, updates=updates)

for _ in range(100):
  my_fn(tf.random.uniform((16,5)))