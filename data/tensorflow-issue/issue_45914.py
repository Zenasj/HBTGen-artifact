# tf.random.uniform((B, T, F, 1), dtype=tf.float32) ← 
# Inferred input shape: B=batch size (dynamic), T=time steps (sequence length), F=feature dims, 1=channel dimension

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, nb_actions=96, hidden_layers=4, layer_neurons=128):
        super().__init__()
        self.nb_actions = nb_actions
        self.hidden_layers = hidden_layers
        self.layer_neurons = layer_neurons

        # Build the two main GRU-based encoders (like main and main2 in the issue)
        # Each produces a feature vector from input sequences 
        # Mimicking build_main(shape) with GRU layers returns final output without sequences
        self.main_grus = []
        for i in range(self.hidden_layers):
            self.main_grus.append(
                tf.keras.layers.GRU(self.layer_neurons, return_sequences=True, name=f"main_GRU{i}")
            )
        # Last GRU layer has return_sequences=False to reduce sequence to single vector
        self.main_gru_final = tf.keras.layers.GRU(self.layer_neurons, return_sequences=False, name=f"main_GRU{self.hidden_layers}")

        self.main2_grus = []
        for i in range(self.hidden_layers):
            self.main2_grus.append(
                tf.keras.layers.GRU(self.layer_neurons, return_sequences=True, name=f"main2_GRU{i}")
            )
        self.main2_gru_final = tf.keras.layers.GRU(self.layer_neurons, return_sequences=False, name=f"main2_GRU{self.hidden_layers}")

        # Inverse model: concatenates the outputs of main and main2,
        # then a Dense layer with sigmoid activation outputs a prediction vector of size nb_actions
        self.icm_inverse_dense = tf.keras.layers.Dense(nb_actions, activation='sigmoid', name='icm_i_output')

        # Forward model: concatenates main output with the action input,
        # then outputs a vector matching the main output feature dimension with a linear Dense layer
        # Output shape matches the main encoder output dimension = layer_neurons
        self.icm_forward_dense = tf.keras.layers.Dense(self.layer_neurons, activation='linear', name='icm_f_output')

    def call(self, inputs, training=False):
        """
        Forward pass expects inputs as tuple:
           (obs1, obs2, icm_action)
           obs1, obs2: sequences of shape (B, T, F, 1) or (B, T, F)
           icm_action: one-hot tensor with shape (B, nb_actions)

        Returns:
          comparison boolean tensor whether inverse_model output matches icm_action within tolerance.

        Note:
          This merges the inverse and forward models with main encoders.
          The original builds use keras.Model separately -
          here fused into a single custom Model per the instructions.
        """

        obs1, obs2, icm_action = inputs

        # Encode obs1 with main GRU stack
        x1 = obs1
        for gru_layer in self.main_grus:
            x1 = gru_layer(x1, training=training)
        x1 = self.main_gru_final(x1, training=training)  # shape (B, layer_neurons)

        # Encode obs2 with main2 GRU stack
        x2 = obs2
        for gru_layer in self.main2_grus:
            x2 = gru_layer(x2, training=training)
        x2 = self.main2_gru_final(x2, training=training)  # shape (B, layer_neurons)

        # Inverse model: concat features from both obs
        inverse_concat = tf.concat([x1, x2], axis=-1)  # shape (B, 2*layer_neurons)
        inverse_pred = self.icm_inverse_dense(inverse_concat)  # shape (B, nb_actions), sigmoid activation

        # Forward model: concat obs1 features and icm action
        forward_concat = tf.concat([x1, icm_action], axis=-1)  # shape (B, layer_neurons + nb_actions)
        forward_pred = self.icm_forward_dense(forward_concat)  # shape (B, layer_neurons), linear activation

        # For comparison, compute if inverse model output matches icm_action with tolerance
        # Using a threshold to binarize inverse_pred since it outputs sigmoid probabilities
        inverse_pred_binary = tf.cast(inverse_pred > 0.5, dtype=tf.float32)
        matches = tf.reduce_all(tf.equal(inverse_pred_binary, icm_action), axis=-1)  # (B,)

        return matches  # boolean tensor shape (B,), True if predictions exactly match ground truth action

def my_model_function():
    # Instantiate MyModel with default parameters from issue context
    return MyModel()

def GetInput():
    # Generate random inputs matching model expectations:
    # obs1 and obs2 are sequences of shape (B, T, F, 1) with dtype float32
    # icm_action: one-hot vector of size nb_actions

    B = 4     # batch size (chosen arbitrarily)
    T = 10    # assumed sequence steps (since original env.shape not fully specified)
    F = 8     # assumed feature dimension (inferred from example reshapes and plausible)
    nb_actions = 96

    # Random float inputs simulating observations
    obs1 = tf.random.uniform((B, T, F, 1), dtype=tf.float32, minval=0.0, maxval=1.0)
    obs2 = tf.random.uniform((B, T, F, 1), dtype=tf.float32, minval=0.0, maxval=1.0)

    # Generate random one-hot actions for icm_action input
    indices = tf.random.uniform((B,), minval=0, maxval=nb_actions, dtype=tf.int32)
    icm_action = tf.one_hot(indices, nb_actions, dtype=tf.float32)

    # The GRU layers expect inputs of shape (B, T, features), so squeeze 4th dim
    # Original code reshapes input to remove leading dims except last 3 possibly
    obs1 = tf.squeeze(obs1, axis=-1)  # (B, T, F)
    obs2 = tf.squeeze(obs2, axis=-1)  # (B, T, F)

    return (obs1, obs2, icm_action)

