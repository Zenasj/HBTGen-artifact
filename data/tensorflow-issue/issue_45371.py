# tf.random.uniform((1, None, 8), dtype=tf.float32) ← input shape: batch_size=1, sequence_length variable, feature_dim=8
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from issue
        self.batch_size = 1
        self.state_dimensionality = 8
        self.n_actions = 4
        self.layer_sizes = (64, 64)
        
        # Build encoder submodel: dense layers with orthogonal init and tanh activation
        encoder_inputs = tf.keras.Input(shape=(self.state_dimensionality,), batch_size=self.batch_size)
        x = encoder_inputs
        for units in self.layer_sizes:
            x = tf.keras.layers.Dense(
                units,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
                bias_initializer=tf.constant_initializer(0.0),
            )(x)
            x = tf.keras.layers.Activation("tanh")(x)
        self.encoder = tf.keras.Model(inputs=encoder_inputs, outputs=x, name="policy_encoder")

        # Masking layer for variable-length sequences
        self.masking = tf.keras.layers.Masking()

        # TimeDistributed wrapper over the encoder to apply on each time step
        self.time_distributed = tf.keras.layers.TimeDistributed(self.encoder, name="TD_policy")

        # Stateful SimpleRNN recurrent layer with orthogonal initialization, returning sequences and states
        self.rnn = tf.keras.layers.SimpleRNN(
            self.layer_sizes[-1],
            stateful=True,
            return_sequences=True,
            return_state=True,
            batch_size=self.batch_size,
            name="policy_recurrent_layer",
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
            bias_initializer=tf.constant_initializer(0.0),
        )

        # Output Dense layer for means of actions
        self.means_dense = tf.keras.layers.Dense(
            self.n_actions,
            name="means",
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
            bias_initializer=tf.keras.initializers.Constant(0.0),
        )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs shape: (batch_size=1, seq_len, state_dimensionality=8)
        x = self.masking(inputs)
        x = self.time_distributed(x)
        # The recurrent layer returns (output_sequence, final_state)
        # We only need the output sequence for downstream dense layer
        x, _ = self.rnn(x)
        means = self.means_dense(x)
        return means

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape: (batch=1, variable seq length, features=8)
    # Using seq length=5 as a reasonable example; dtype float32
    return tf.random.uniform((1, 5, 8), dtype=tf.float32)

