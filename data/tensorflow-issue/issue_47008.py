# tf.random.uniform((B, None, 2), dtype=tf.float32)  # Input shape: variable length time series with 2 features per step

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Masking, GRU, TimeDistributed, Dense, Lambda
from tensorflow.keras.optimizers import Adam

# The original code was using a custom WTTE-RNN layer outputs and loss.
# Since those functions (wtte.output_lambda and wtte.loss) are from a separate package
# and not provided, we'll create reasonable placeholders for those.

# Placeholder for the WTTE output lambda function
def wtte_output_lambda(x, init_alpha=47.05906039370126, max_beta_value=4.0, alpha_kernel_scalefactor=0.5):
    # In the original repo, this probably transforms the 2-channel output into parameters of a distribution.
    # Here we just clip/scale outputs as a placeholder.
    # Apply a positive activation for alpha param and bound beta param.
    alpha = tf.nn.softplus(x[..., 0:1]) * alpha_kernel_scalefactor + init_alpha
    beta = tf.clip_by_value(tf.nn.softplus(x[..., 1:2]), 1e-5, max_beta_value)
    return tf.concat([alpha, beta], axis=-1)

# Wrap as a Keras Lambda layer compatible function
def wtte_output_lambda_layer(init_alpha=47.05906039370126,
                             max_beta_value=4.0,
                             alpha_kernel_scalefactor=0.5):
    return Lambda(
        lambda x: wtte_output_lambda(
            x,
            init_alpha=init_alpha,
            max_beta_value=max_beta_value,
            alpha_kernel_scalefactor=alpha_kernel_scalefactor
        )
    )


# Placeholder for the WTTE loss function
# The original loss is a discrete version of Weibull Time-To-Event loss computed per timestep.
# We create a dummy loss that acts on the predicted alpha/beta params and true values.
# This will just combine a simple negative log likelihood inspired loss (just for demonstration).
def wtte_loss(kind='discrete', reduce_loss=False):
    def loss_function(y_true, y_pred):
        # y_pred shape: (batch, time, 2) - predicted alpha, beta
        # y_true shape: (batch, time, *)
        # For placeholder, treat y_true[...,0] as "time" and y_true[...,1] as event indicator (dummy)
        alpha = y_pred[..., 0]
        beta = y_pred[..., 1]
        t = y_true[..., 0]
        event = y_true[..., 1]

        # Add small epsilon for numerical stability
        eps = 1e-6

        # Weibull PDF (discrete version placeholder)
        hazard0 = (t / (beta + eps)) ** (alpha - 1)
        likelihood = hazard0 * tf.exp(-(t / (beta + eps)) ** alpha)
        event_loss = -tf.math.log(likelihood + eps) * event
        censored_loss = (t / (beta + eps)) ** alpha * (1 - event)

        loss_raw = event_loss + censored_loss

        if reduce_loss:
            return tf.reduce_mean(loss_raw)
        else:
            return loss_raw

    return type('LossWrapper', (), {'loss_function': loss_function})()


class MyModel(tf.keras.Model):
    def __init__(self, mask_value=-1.3371337, 
                 init_alpha=47.05906039370126,
                 max_beta_value=4.0,
                 alpha_kernel_scalefactor=0.5,
                 gru_units=3):
        super().__init__()
        # Masking layer to skip timesteps with mask value
        self.mask = Masking(mask_value=mask_value, input_shape=(None, 2))
        # GRU layer (return sequences)
        self.gru = GRU(gru_units, activation='tanh', return_sequences=True)
        # TimeDistributed dense to output 2 params per timestep
        self.time_dense = TimeDistributed(Dense(2))
        # Lambda layer applying wtte output transform
        self.output_lambda = Lambda(
            lambda x: wtte_output_lambda(
                x,
                init_alpha=init_alpha,
                max_beta_value=max_beta_value,
                alpha_kernel_scalefactor=alpha_kernel_scalefactor
            )
        )

        # Store loss function for use in compile
        self.loss_fn = wtte_loss(kind='discrete', reduce_loss=False).loss_function

        # Compile the model here (with optimizer and loss)
        self.compile(
            optimizer=Adam(learning_rate=0.01, clipvalue=0.5),
            loss=self.loss_fn,
            sample_weight_mode='temporal'
        )

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        x = self.mask(inputs)
        x = self.gru(x)
        x = self.time_dense(x)
        x = self.output_lambda(x)
        return x


def my_model_function():
    return MyModel()


def GetInput():
    # Generate a random batch of data to simulate input:
    # As per the original input shape:
    # batch_size: 5 (common from the user's generator batch size)
    # time steps: variable length, let's choose 50 for example
    # features: 2 (matches input_shape=(None, 2))
    batch_size = 5
    time_steps = 50
    features = 2

    # Use tf.random.uniform with standard float32 dtype
    return tf.random.uniform(
        shape=(batch_size, time_steps, features), 
        minval=0.0, maxval=1.0, dtype=tf.float32
    )

