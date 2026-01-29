# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_model_optimization as tfmot

# A No-Op QuantizeConfig to allow annotating BatchNormalization layers without actual quantization
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

quantize_config = NoOpQuantizeConfig()

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define LeNet part
        self.lenet_padding = layers.ZeroPadding2D(padding=(2, 2))
        self.lenet_conv1 = layers.Conv2D(6, 5, strides=(1, 1), activation='relu')
        self.lenet_bn1 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.lenet_pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.lenet_conv2 = layers.Conv2D(16, 5, strides=(1, 1), activation='relu')
        self.lenet_bn2 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.lenet_pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.lenet_flatten = layers.Flatten()
        self.lenet_dense1 = layers.Dense(120, activation='relu')
        self.lenet_bn3 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.lenet_dense2 = layers.Dense(84, activation='relu')
        self.lenet_bn4 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.lenet_output = layers.Dense(10, activation='softmax')

        # Define XMLP part
        self.xmlp_flatten = layers.Flatten()
        self.xmlp_dense1 = layers.Dense(512, activation='linear')
        self.xmlp_bn1 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.xmlp_dense2 = layers.Dense(512, activation='relu')
        self.xmlp_dense3 = layers.Dense(512, activation='linear')
        self.xmlp_bn2 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.xmlp_dense4 = layers.Dense(512, activation='relu')
        self.xmlp_dense5 = layers.Dense(512, activation='linear')
        self.xmlp_bn3 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.xmlp_dense6 = layers.Dense(512, activation='relu')
        self.xmlp_dense7 = layers.Dense(256, activation='linear')
        self.xmlp_bn4 = tfmot.quantization.keras.quantize_annotate_layer(
            layers.BatchNormalization(), quantize_config=quantize_config)
        self.xmlp_dense8 = layers.Dense(256, activation='relu')
        self.xmlp_output = layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # LeNet forward
        x = self.lenet_padding(inputs)
        x = self.lenet_conv1(x)
        x = self.lenet_bn1(x, training=training)
        x = self.lenet_pool1(x)
        x = self.lenet_conv2(x)
        x = self.lenet_bn2(x, training=training)
        x = self.lenet_pool2(x)
        x = self.lenet_flatten(x)
        x = self.lenet_dense1(x)
        x = self.lenet_bn3(x, training=training)
        x = self.lenet_dense2(x)
        x = self.lenet_bn4(x, training=training)
        lenet_out = self.lenet_output(x)

        # XMLP forward
        y = self.xmlp_flatten(inputs)
        y = self.xmlp_dense1(y)
        y = self.xmlp_bn1(y, training=training)
        y = self.xmlp_dense2(y)
        y = self.xmlp_dense3(y)
        y = self.xmlp_bn2(y, training=training)
        y = self.xmlp_dense4(y)
        y = self.xmlp_dense5(y)
        y = self.xmlp_bn3(y, training=training)
        y = self.xmlp_dense6(y)
        y = self.xmlp_dense7(y)
        y = self.xmlp_bn4(y, training=training)
        y = self.xmlp_dense8(y)
        xmlp_out = self.xmlp_output(y)

        # For demonstration, output a dictionary with both outputs
        # One could implement comparison or fusion logic here if needed
        return {'LeNet': lenet_out, 'XMLP': xmlp_out}

def my_model_function():
    # Create an instance of MyModel composed of two submodels from the issue.
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch, height, width, channels)
    # MNIST images are (28,28,1) grayscale images.
    batch_size = 4  # Can be any batch size
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32, minval=0.0, maxval=1.0)

