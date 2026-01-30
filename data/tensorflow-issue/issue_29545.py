import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

class resblock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons,
                                          activation='elu',
                                          kernel_initializer='he_normal')
                       for _ in range(n_layers)]
        
    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return inputs + z
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class res_mod(keras.models.Model):
    def __init__(self, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.f1 = keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal')
        self.b1 = resblock(2, 100)
        self.b2 = resblock(2, 100)
        self.output1 = keras.layers.Dense(output_dim, activation=keras.activations.get(activation))
        
    def call(self, inputs):
        z = self.f1(inputs)
        z = self.hidden1(z)
        for _ in range(4):
            z = self.b1(z)
        z = self.b2(z)
        return self.output1(z)
    
    def get_config(self):
        base_config = super().get_config()
        return{**base_config, "output_dim" : output_dim, "activation": activation}
    

model = res_mod(10, activation='softmax')
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(train, epochs=25, validation_data=test)

# This is able to save and works correctly, returning the trained model
model.save('custom_model.h5py')
del model
model = keras.models.load_model('custom_model.h5py', custom_objects={'resblock': resblock})

model.save('custom_model.h5')

x = tf.random.uniform((100,))
y = tf.random.uniform((100,))


class test_model(keras.models.Model):
    def __init__(self, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.f1 = keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal', name="h1")
        self.output1 = keras.layers.Dense(output_dim, activation=keras.activations.get(activation))
        
    def call(self, inputs):
        z = self.f1(inputs)
        z = self.hidden1(z)
        return self.output1(z)
    
    def get_config(self):
        base_config = super().get_config()
        return{**base_config, "output_dim" : output_dim, "activation": activation}
    
model = test_model(1)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=5)
print(model.weights[0])
model.save('custom_model.hdf5')
del model
model = keras.models.load_model('custom_model.hdf5')
print(model.weights[0])

x = tf.random.uniform((100,))
y = tf.random.uniform((100,))


class test_model(keras.models.Model):
    def __init__(self, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.f1 = keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal', name="h1")
        self.output1 = keras.layers.Dense(output_dim, activation=keras.activations.get(activation))
        
    def call(self, inputs):
        z = self.f1(inputs)
        z = self.hidden1(z)
        return self.output1(z)
    
    def get_config(self):
        base_config = super().get_config()
        return{**base_config, "output_dim" : output_dim, "activation": activation}
    
model = test_model(1)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=5)
print(model.weights[0])
model.save('custom_model.h5py')
del model
model = keras.models.load_model('custom_model.h5py')
print(model.weights[0])

model.save("NameOfModel", save_format='tf')

loaded_tfkmodel = tf.keras.models.load_model('./NameOfModel')

keras.models.load_model('path_to_my_model')

model.save_weights('model_weights', save_format='tf')

loaded_model = ClassifierModel(parameter)
loaded_model.compile(parameters)
loaded_model.train_on_batch(x_train[:1], y_train[:1])
loaded_model.load_weights('model_weights')