import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution() # comment to enable eager execution mode
print("[INFO] Eager mode: ", tf.executing_eagerly()) # For easy reset of notebook state.

class CustomModelV2(tf.keras.Model):
    def __init__(self):
        super(CustomModelV2, self).__init__()
        self.encoder = Encoder(32)
        self.encoder.build(input_shape=(None, 32))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def call(self, inputs, training):
        return self.encoder(inputs, training)
        
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]

    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            r_loss = tf.keras.losses.mean_squared_error(y, y_pred)
            loss = r_loss 

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

class Encoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Encoder, self).__init__(name = 'Encoder')
        self.input_layer   = DenseLayer(128, input_size, 0.0, 0.0, 'float32')
        self.hidden_layer1 = DenseLayer(128, 128, 0.001, 0.0, 'float32')
        self.dropout_laye1 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer2 = DenseLayer(64, 128, 0.001, 0.0, 'float32')      
        self.dropout_laye2 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer3 = DenseLayer(64, 64, 0.001, 0.0, 'float32')
        self.dropout_laye3 = tf.keras.layers.Dropout(0.2)           
        self.output_layer  = LinearLayer(64, 64, 0.001, 0.0, 'float32')
        
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

class LinearLayer(tf.keras.layers.Layer):

    def __init__(self, units, input_dim, weights_regularizer, bias_regularizer, d_type):
        super(LinearLayer, self).__init__()
        self.w = self.add_weight(name='w_linear',
                                shape = (input_dim, units), 
                                initializer = tf.keras.initializers.RandomUniform(
                                    minval=-tf.cast(tf.math.sqrt(6/(input_dim+units)), dtype = d_type), 
                                    maxval=tf.cast(tf.math.sqrt(6/(input_dim+units)), dtype = d_type), 
                                    seed=16751),                                                                   
                                regularizer = tf.keras.regularizers.l1(weights_regularizer), 
                                trainable = True)
        self.b = self.add_weight(name='b_linear',
                                 shape = (units,),    
                                 initializer = tf.zeros_initializer(),
                                 regularizer = tf.keras.regularizers.l1(bias_regularizer),
                                 trainable = True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, units, input_dim, weights_regularizer, bias_regularizer, d_type):
        super(DenseLayer, self).__init__()
        self.w = self.add_weight(name='w_dense',
                                 shape = (input_dim, units), 
                                 initializer = tf.keras.initializers.RandomUniform(
                                     minval=-tf.cast(tf.math.sqrt(6.0/(input_dim+units)), dtype = d_type),  
                                     maxval=tf.cast(tf.math.sqrt(6.0/(input_dim+units)), dtype = d_type),  
                                     seed=16751), 
                                 regularizer = tf.keras.regularizers.l1(weights_regularizer), 
                                 trainable = True)
        self.b = self.add_weight(name='b_dense',
                                 shape = (units,),    
                                 initializer = tf.zeros_initializer(),
                                 regularizer = tf.keras.regularizers.l1(bias_regularizer),
                                 trainable = True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return tf.nn.elu(x)

# Just use `fit` as usual
x = tf.data.Dataset.from_tensor_slices(np.random.random((5000, 32)))

y_numpy = np.random.random((5000, 1))
y_numpy[:, 3:] = None
y = tf.data.Dataset.from_tensor_slices(y_numpy)

x_window = x.window(30, shift=10, stride=1)
flat_x = x_window.flat_map(lambda t: t)
flat_x_scaled = flat_x.map(lambda t: t * 2)

y_window = y.window(30, shift=10, stride=1)
flat_y = y_window.flat_map(lambda t: t)
flat_y_scaled = flat_y.map(lambda t: t * 2)

z = tf.data.Dataset.zip((flat_x_scaled, flat_y_scaled)).batch(32).cache().shuffle(buffer_size=32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Stopping criteria if the training loss doesn't go down by 1e-3
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta = 1e-3, verbose = 1, mode='min', patience = 3, 
    baseline=None, restore_best_weights=True)

# Construct and compile an instance of CustomModel
model = CustomModelV2()


  
model.compile(optimizer=tf.optimizers.Adagrad(0.01))

history = model.fit(z, epochs=3, callbacks=[early_stop_cb])