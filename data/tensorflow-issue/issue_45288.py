import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def init_networks():
    # Inputs
    state_input = tf.keras.Input(shape=(d), name='state_input')
    action_input = tf.keras.Input(shape=(), dtype=tf.int32, name='action_input')
    
    # Layers
    hidden_layers = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dense(nA),
    ], name='hidden_layers')
    
    # Outputs
    hidden_output = hidden_layers(state_input)

    # Models
    hidden_net = tf.keras.Model(inputs=[state_input], outputs=[hidden_output], name='hidden_net')
    
    return hidden_net

hidden_net = init_networks()