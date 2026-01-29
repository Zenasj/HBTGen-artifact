# tf.random.uniform((B, 4), dtype=tf.float32)  # Input shape corresponds to state vector from AcrobotForever-v1 (4 state variables)

import tensorflow as tf
from tensorflow.keras import layers, Model

class MyModel(tf.keras.Model):
    def __init__(self, ob_dim=4, n_actions=3):
        """
        A TensorFlow 2.x reimplementation of the Keras Sequential model used in ModelBasedAgent,
        adapted to subclass tf.keras.Model and use functional calls.
        
        ob_dim: dimensionality of state input (4 for AcrobotForever-v1)
        n_actions: discrete action space size (3 for Acrobot)
        
        The model predicts delta state (state_next - state) from concatenated (state, action).
        """
        super().__init__()
        
        self.ob_dim = ob_dim
        self.n_actions = n_actions
        
        # Model layers for (state + action) -> delta_state prediction
        self.dense1 = layers.Dense(12, activation='relu', input_shape=(ob_dim + 1,))
        self.dense2 = layers.Dense(12, activation='relu')
        self.dense3 = layers.Dense(ob_dim, activation='linear')
        
    def call(self, inputs, training=False):
        """
        inputs: Tensor of shape (batch_size, ob_dim + 1)
          concatenation of state (4 floats) and action (1 float)
          
        returns:
          predicted delta state (batch_size, ob_dim)
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        delta_state = self.dense3(x)
        return delta_state
    
    def predict_delta_state(self, state, action):
        """
        Convenience function to predict delta state from single state + action inputs.
        state: Tensor of shape (ob_dim,)
        action: scalar or tensor of shape ()
        
        Returns: Tensor of shape (ob_dim,)
        """
        sa = tf.concat([state, tf.cast(tf.reshape(action, (1,)), tf.float32)], axis=0)
        sa = tf.expand_dims(sa, axis=0)  # add batch dimension: (1, ob_dim+1)
        ds = self(sa, training=False)
        return tf.squeeze(ds, axis=0)  # shape (ob_dim,)

def my_model_function():
    model = MyModel(ob_dim=4, n_actions=3)
    # Compile model with optimizer and loss for convenient training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

def GetInput():
    """
    Return a random tensor input suitable for MyModel call.
    Input shape must be (batch_size, 5), where 5 = 4 state dims + 1 action dim.
    
    We'll create batch_size = 1, with values in typical normalized range.
    Assume state values roughly between -1.0 and 1.0 (angles and velocities),
    and action as discrete integer cast to float (between 0 and n_actions-1).
    """
    import numpy as np
    state = np.random.uniform(low=-1.0, high=1.0, size=(1, 4)).astype(np.float32)
    action = np.random.randint(low=0, high=3, size=(1,1)).astype(np.float32)
    input_tensor = tf.constant(np.concatenate([state, action], axis=-1))
    return input_tensor

