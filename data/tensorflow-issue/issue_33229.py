import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    @tf.function
    def call(self, y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        is_small_error = error <= self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = error * self.threshold - 0.5 * self.threshold**2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        cfg = super().get_config()
        cfg['threshold'] = self.threshold
        return cfg

model = keras.Sequential([
  keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
  keras.layers.Dense(1)
])

model.compile(loss=HuberLoss(2.0), optimizer="sgd")
model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))

# save and reload
model.save('model_with_huber_loss_class.h5')
model = keras.models.load_model('model_with_huber_loss_class.h5', custom_objects={'HuberLoss': HuberLoss})

# continue training
model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))

if isinstance(loss_config, dict) and 'class_name' in loss_config:
    loss = losses.deserialize(loss_config, custom_objects)  # <<< this is executed

if True:
    # LOSS custom class: FIXED by the PR (thanks @omalleyt12)
    model.compile(loss=HuberLoss(2.0), optimizer="sgd")
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))
    model.save('model.h5')
    model = keras.models.load_model('model.h5', custom_objects={'HuberLoss': HuberLoss})

if False:
    # LOSS custom function: was working before the PR, still works with the PR
    model.compile(loss=huber_fn, optimizer="sgd")
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))
    model.save('model.h5')
    model = keras.models.load_model('model.h5', custom_objects={'huber_fn': huber_fn})

if False:
    # METRICS custom class: was broken before the PR, still broken with the PR
    model.compile(loss=keras.losses.mean_squared_error, optimizer="sgd", metrics=[HuberMetric(threshold=.2)])
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))
    model.save('model.h5')
    model = keras.models.load_model('model.h5', custom_objects={'HuberMetric':HuberMetric})

if False:
    # METRICS custom function: was working before the PR, still works with the PR
    model.compile(loss=keras.losses.mean_squared_error, optimizer="sgd", metrics=[huber_fn])
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))
    model.save('model.h5')
    model = keras.models.load_model('model.h5', custom_objects={'huber_fn':huber_fn})

# continue training with the reloaded model
model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))