from tensorflow import keras
from tensorflow.keras import layers

class ThreeLayerMLP(keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = {"hidden_units": self.hidden_units}
        return config

model = ThreeLayerMLP([64, 64, 10])
serialized_model = keras.layers.serialize(model)
new_model = keras.layers.deserialize(
    serialized_model, custom_objects={'ThreeLayerMLP': ThreeLayerMLP})