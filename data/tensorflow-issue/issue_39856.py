from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

#%%
import numpy as np
import tensorflow as tf
print(tf.__version__)

#%% 
class Composite(tf.keras.Model):
    def __init__(self, *args, **kwargs):

        super(Composite, self).__init__(*args, **kwargs)

    def train_step(self, data):

        data_adapter = tf.python.keras.engine.data_adapter
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        tf.print("HIHI! I'm in function train_step!")

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        _minimize = tf.python.keras.engine.training._minimize
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

in_ = tf.keras.layers.Input(shape=(10, ) )
x = tf.keras.layers.Dense(1)(in_)
model = Composite(inputs=in_, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='SGD', metrics=['accuracy'])

X = np.zeros((10,10))
Y = np.zeros((10,1))
model.fit(X,Y,verbose=2)

# %%
new_model = tf.keras.models.clone_model(model)
new_model.compile(loss='binary_crossentropy',optimizer='SGD', metrics=['accuracy'])
new_model.fit(X,Y,verbose=2)

wrap_model = Composite(inputs=new_model.input, outputs=new_model.output) 
wrap_model.compile(loss='binary_crossentropy',optimizer='SGD', metrics=['accuracy'])
wrap_model.fit(X,Y,verbose=2)

#%%
import numpy as np
import tensorflow as tf
print(tf.__version__)

#%% 
class Composite(tf.keras.Model):
    def __init__(self, *args, **kwargs):

        super(Composite, self).__init__(*args, **kwargs)

    def unpack_x_y_sample_weight(self, data):
        """Unpacks user-provided data tuple."""
        if not isinstance(data, tuple):
            return (data, None, None)
        elif len(data) == 1:
            return (data[0], None, None)
        elif len(data) == 2:
            return (data[0], data[1], None)
        elif len(data) == 3:
            return (data[0], data[1], data[2])

    def train_step(self, data):

        x, y, sample_weight = self.unpack_x_y_sample_weight(data)

        tf.print("HIHI! I'm in function train_step!")

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

in_ = tf.keras.layers.Input(shape=(10, ) )
x = tf.keras.layers.Dense(1)(in_)
model = Composite(inputs=in_, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='SGD', metrics=['accuracy'])

X = np.zeros((10,10))
Y = np.zeros((10,1))

print("the original model")
model.fit(X,Y,verbose=2)

# %%
print("cloned model")
new_model = tf.keras.models.clone_model(model)
new_model.compile(loss='binary_crossentropy',optimizer='SGD', metrics=['accuracy'])
new_model.fit(X,Y,verbose=2)

print("the original model again")
model.fit(X,Y,verbose=2)