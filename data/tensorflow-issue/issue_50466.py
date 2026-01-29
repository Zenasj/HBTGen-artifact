# tf.random.uniform((batch_size, 32), dtype=tf.float32)
import tensorflow as tf

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, weights_regularizer, bias_regularizer, d_type):
        super(DenseLayer, self).__init__()
        self.w = self.add_weight(name='w_dense',
                                 shape=(input_dim, units),
                                 initializer=tf.keras.initializers.RandomUniform(
                                     minval=-tf.cast(tf.math.sqrt(6.0/(input_dim+units)), dtype=d_type),
                                     maxval=tf.cast(tf.math.sqrt(6.0/(input_dim+units)), dtype=d_type),
                                     seed=16751),
                                 regularizer=tf.keras.regularizers.l1(weights_regularizer),
                                 trainable=True)
        self.b = self.add_weight(name='b_dense',
                                 shape=(units,),
                                 initializer=tf.zeros_initializer(),
                                 regularizer=tf.keras.regularizers.l1(bias_regularizer),
                                 trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return tf.nn.elu(x)


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, weights_regularizer, bias_regularizer, d_type):
        super(LinearLayer, self).__init__()
        self.w = self.add_weight(name='w_linear',
                                 shape=(input_dim, units),
                                 initializer=tf.keras.initializers.RandomUniform(
                                     minval=-tf.cast(tf.math.sqrt(6/(input_dim + units)), dtype=d_type),
                                     maxval=tf.cast(tf.math.sqrt(6/(input_dim + units)), dtype=d_type),
                                     seed=16751),
                                 regularizer=tf.keras.regularizers.l1(weights_regularizer),
                                 trainable=True)
        self.b = self.add_weight(name='b_linear',
                                 shape=(units,),
                                 initializer=tf.zeros_initializer(),
                                 regularizer=tf.keras.regularizers.l1(bias_regularizer),
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Encoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Encoder, self).__init__(name='Encoder')
        self.input_layer = DenseLayer(128, input_size, 0.0, 0.0, 'float32')
        self.hidden_layer1 = DenseLayer(128, 128, 0.001, 0.0, 'float32')
        self.dropout_laye1 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer2 = DenseLayer(64, 128, 0.001, 0.0, 'float32')
        self.dropout_laye2 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer3 = DenseLayer(64, 64, 0.001, 0.0, 'float32')
        self.dropout_laye3 = tf.keras.layers.Dropout(0.2)
        self.output_layer = LinearLayer(64, 64, 0.001, 0.0, 'float32')

    def call(self, input_data, training):
        fx = self.input_layer(input_data)
        fx = self.hidden_layer1(fx)
        if training:
            fx = self.dropout_laye1(fx)
        fx = self.hidden_layer2(fx)
        if training:
            fx = self.dropout_laye2(fx)
        fx = self.hidden_layer3(fx)
        if training:
            fx = self.dropout_laye3(fx)
        return self.output_layer(fx)


class MyModel(tf.keras.Model):
    """
    This model corresponds to the original 'CustomModelV2' in the issue.
    It wraps an Encoder instance and implements custom training logic.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = Encoder(32)
        # Build allow weights creation based on input shape.
        self.encoder.build(input_shape=(None, 32))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.encoder(inputs, training)

    @property
    def metrics(self):
        # So that reset_states() can be called automatically by fit/evaluate
        return [self.loss_tracker]

    @tf.function
    def train_step(self, data):
        # Unpack inputs and targets
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)
            # Mean squared error (elementwise mean) loss
            r_loss = tf.keras.losses.mean_squared_error(y, y_pred)
            # mean_squared_error returns per-sample loss, so reduce mean here
            loss = tf.reduce_mean(r_loss)

            # Add any additional losses, e.g. from regularizers in layers
            if self.losses:
                loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}


def my_model_function():
    # Instantiate and compile the model similarly to the original example
    model = MyModel()
    model.compile(optimizer=tf.optimizers.Adagrad(0.01))
    return model


def GetInput():
    # Based on the build/input_shape of the Encoder and model,
    # input shape is (batch_size, 32) with float32 dtype
    batch_size = 32  # typical batch size
    return tf.random.uniform((batch_size, 32), dtype=tf.float32)

