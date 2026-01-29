# tf.random.uniform((B, None, 1), dtype=tf.float32) ← Input shape is variable-length sequences with 1 feature per timestep

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        self.gru = tf.keras.layers.GRU(units=3, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=1)
        # Mean squared error loss and metric
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.metric_fn = tf.keras.metrics.MeanSquaredError()
    
    def call(self, inputs, training=None):
        # inputs is expected to be shape (batch, timesteps, 1)
        # Apply masking (mask zeros)
        masked = self.masking(inputs)
        x = self.gru(masked)
        output = self.dense(x)
        return output
    
    def compute_masked_metrics_and_loss(self, y_true, y_pred, mask):
        # y_true, y_pred shape: (batch, timesteps, 1)
        # mask shape: (batch, timesteps)
        
        # Convert mask (bool) to float32
        mask_f = tf.cast(mask, tf.float32)
        # Compute elementwise squared error
        se = tf.square(y_true - y_pred)  # shape: (batch, timesteps, 1)
        se = tf.squeeze(se, axis=-1)     # shape: (batch, timesteps)
        
        # Masked sum of squared errors and count of unmasked values
        masked_se_sum = tf.reduce_sum(se * mask_f)  # sum only unmasked steps
        valid_count = tf.reduce_sum(mask_f)
        
        # Safe division to avoid div-by-zero
        loss = tf.math.divide_no_nan(masked_se_sum, valid_count)
        
        # Metric here computed with same masking logic
        self.metric_fn.reset_state()
        # Flatten batch+time dimension where mask is True
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        self.metric_fn.update_state(y_true_masked, y_pred_masked)
        metric = self.metric_fn.result()
        return loss, metric

    def train_step(self, data):
        x, y = data
        # Obtain mask from input (mask value 0.0)
        mask = tf.not_equal(tf.reduce_sum(tf.abs(x), axis=-1), 0)
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss, metric = self.compute_masked_metrics_and_loss(y, y_pred, mask)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics state to the current metric value computed
        return {"loss": loss, "mean_squared_error": metric}
    
    def test_step(self, data):
        x, y = data
        mask = tf.not_equal(tf.reduce_sum(tf.abs(x), axis=-1), 0)
        y_pred = self(x, training=False)
        loss, metric = self.compute_masked_metrics_and_loss(y, y_pred, mask)
        return {"loss": loss, "mean_squared_error": metric}


def my_model_function():
    model = MyModel()
    # Compile model with optimizer,
    # but don't pass in loss/metrics because overridden train_step/test_step handle them
    model.compile(optimizer=tf.keras.optimizers.RMSprop())
    return model


def GetInput():
    # Generate a random batch of variable-length padded sequences with padding 0
    # Assuming batch size 10, max length 20, 1 feature per timestep
    batch_size = 10
    max_length = 20
    feature_dim = 1
    # Create variable sequence lengths between 5 and max_length
    lengths = tf.random.uniform((batch_size,), minval=5, maxval=max_length+1, dtype=tf.int32)
    inputs = tf.zeros((batch_size, max_length, feature_dim), dtype=tf.float32)
    for i in range(batch_size):
        seq_len = lengths[i]
        # Fill with random sine waves for realistic values (simulate input)
        t = tf.linspace(0.0, 3.14, seq_len)
        sine_wave = tf.expand_dims(tf.sin(t), axis=-1)
        # Pad to max_length with zeros automatically
        paddings = [[0, max_length - seq_len], [0, 0]]
        padded_seq = tf.pad(sine_wave, paddings)
        inputs = tf.tensor_scatter_nd_update(inputs, [[i]], [padded_seq])
    return inputs

