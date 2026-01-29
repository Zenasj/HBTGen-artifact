# tf.random.uniform((batch, 1, 5, 5, channels), dtype=tf.float32) ← inferred input shapes (batch, 1, 5, 5, channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define inputs with shape (1, 5, 5, C) for RGB (3 channels), gray (1 channel), and mix (1 channel)
        # The batch dimension is omitted in Input layers
        self.input_rgb = tf.keras.layers.Input(shape=(1,5,5,3), name="input_rgb")
        self.input_gray = tf.keras.layers.Input(shape=(1,5,5,1), name="input_gray")
        self.input_mix = tf.keras.layers.Input(shape=(1,5,5,1), name="input_mix")

        # Concatenate inputs along the channel dimension, axis = -1
        # Because each input shape is (1,5,5,C), axis -1 corresponds to channels
        self.concat = tf.keras.layers.Concatenate(name="rbg_gray")

        # Dense layers are used after flattening the concatenated tensor.
        # Since Dense works on last dimension, flatten first.
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers as in original code:
        self.dense1 = tf.keras.layers.Dense(1, activation='relu',name="Dense_1")
        self.dense2 = tf.keras.layers.Dense(1, activation='softmax',name="softmax")

    def call(self, inputs, training=False):
        # inputs: tuple/list of three tensors (input_rgb, input_gray, input_mix)
        input_rgb, input_gray, input_mix = inputs
        # Concatenate along channels axis (-1)
        concat_inputs = self.concat([input_rgb, input_gray, input_mix])  # shape (..., channels_concat)
        flat = self.flatten(concat_inputs)
        x = self.dense1(flat)
        x = self.dense2(x)
        return x

def my_model_function():
    # Create functional Keras model with same inputs and outputs to allow Keras features
    input_rgb = tf.keras.layers.Input(shape=(1,5,5,3), name="input_rgb")
    input_gray = tf.keras.layers.Input(shape=(1,5,5,1), name="input_gray")
    input_mix = tf.keras.layers.Input(shape=(1,5,5,1), name="input_mix")

    concat = tf.keras.layers.Concatenate(name="rbg_gray")([input_rgb, input_gray, input_mix])
    flat = tf.keras.layers.Flatten()(concat)
    dense1 = tf.keras.layers.Dense(1, activation='relu', name='Dense_1')(flat)
    output = tf.keras.layers.Dense(1, activation='softmax', name='softmax')(dense1)

    keras_model = tf.keras.Model(inputs=[input_rgb, input_gray, input_mix], outputs=output)
    keras_model.compile(loss={ 'softmax': 'binary_crossentropy'}, optimizer=tf.keras.optimizers.Adam())

    # Instantiate MyModel and copy weights from keras_model to MyModel
    my_model = MyModel()
    # To copy weights, we build the model by calling once
    dummy_input = GetInput()
    my_model(dummy_input)
    # Map layers by name to copy weights
    # Note: MyModel layers created inside __init__ and called via call()
    # Keras functional model layers can be accessed by name as well
    layer_map = {layer.name:layer for layer in keras_model.layers}
    for layer in my_model.layers:
        if layer.name in layer_map:
            layer.set_weights(layer_map[layer.name].get_weights())

    return my_model

def GetInput():
    # Return a 3-tuple of random tensors matching input shapes:
    # Shapes (batch_size, 1, 5, 5, channels)
    batch_size = 2  # arbitrary batch size for testing
    input_rgb = tf.random.uniform((batch_size,1,5,5,3), dtype=tf.float32)
    input_gray = tf.random.uniform((batch_size,1,5,5,1), dtype=tf.float32)
    input_mix = tf.random.uniform((batch_size,1,5,5,1), dtype=tf.float32)
    return (input_rgb, input_gray, input_mix)

