from tensorflow.keras import layers
from tensorflow.keras import optimizers

3
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

# Create shared model.
shared_input = Input((10,))
layer = Dense(64, activation='relu')(shared_input)
layer = Dense(128, activation='relu')(layer)
shared_model = Model(shared_input, layer)

# Create actor model on top of shared model.
actor_layer = Dense(128, activation='relu')(shared_model.output)
actor_output = Dense(5, activation='softmax')(actor_layer)
actor_model = Model(shared_model.input, actor_output)

# Create critic model on top of shared model.
critic_layer = Dense(128, activation='relu')(shared_model.output)
critic_output = Dense(1, activation='linear')(critic_layer)
critic_model = Model(shared_model.input, critic_output)

# Create actor optimizer
action_pl = K.placeholder(shape=(None, 5))
advantages_pl = K.placeholder(shape=(None,))
weighted_actions = K.sum(action_pl * actor_model.output, axis=1)
eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages_pl)
entropy = K.sum(actor_model.output * K.log(actor_model.output + 1e-10), axis=1)
loss = 0.001 * entropy - K.sum(eligibility)
# TypeError: get_updates() takes 3 positional arguments but 4 were given.
updates = RMSprop(lr=0.0001, epsilon=0.1, rho=0.99).get_updates(actor_model.trainable_weights, [], loss)
actor_opt = K.function([actor_model.input, action_pl, advantages_pl], [], updates=updates)

3
get_updates(actor_model.trainable_weights, [], loss)

3
get_updates(params=actor_model.trainable_weights, loss=loss)