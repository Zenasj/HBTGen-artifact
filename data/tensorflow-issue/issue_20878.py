# tf.random.uniform((1, 128, 9, 1), dtype=tf.float32)  # inferred input shape from the example sequential model toward the end

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    """
    This fused model includes two submodels discussed in the issue:
    1) The larger MobileNetV2-style model for image input (224,224,3) with 10-class output (from chunk 8-18).
       This is reconstructed partly as a baseline stub due to complexity and incomplete code,
       serving as a representative rather than a full exact topology.
    2) The example smaller Keras Conv2D model for shape (128,9,1) with 6 sigmoid outputs from the final chunks 25-26.

    The call method accepts a tuple input (image_input, small_input),
    and returns a dict with the outputs of each submodel for comparison or usage.

    Assumptions and notes:
    - We provide two dummy submodels reconstructed from descriptions.
    - The image Input shape is assumed (224,224,3) or (150,150,3) depending on usage.
      For simplicity here, the image model uses (224,224,3).
    - The smaller Conv model uses input (128,9,1).
    - The call output returns a dict with both outputs.
    - This fusion reflects discussion of multiple models in the issue.
    """

    def __init__(self):
        super().__init__()

        # --------- Submodel 1: Simplified MobileNetV2-like convolutional backbone ------------
        # Due to complexity, this is a very simplified stub capturing the essence:
        # Conv2D -> BatchNorm -> ReLU6 (to show custom relu6 usage) -> Depthwise separable conv blocks simplified
        def relu6(x):
            return tf.keras.activations.relu(x, max_value=6)

        inputs_img = keras.Input(shape=(224, 224, 3), name="image_input")

        x = keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inputs_img)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(relu6)(x)

        # Simplified depthwise separable conv blocks (instead of detailed blocks in original)
        x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(relu6)(x)
        x = keras.layers.Conv2D(16, 1, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)

        # Global pooling and dense for 10-class output as per 'distribution' Dense layer in chunks
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        out_img = keras.layers.Dense(10, activation="softmax", name="distribution")(x)

        self.submodel_img = keras.Model(inputs=inputs_img, outputs=out_img, name="image_model")

        # --------- Submodel 2: Smaller Conv2D model from chunk 25/26 for sequence input ---------
        # Input shape (128,9,1), output 6 sigmoid classes
        inputs_seq = keras.Input(shape=(128, 9, 1), name="seq_input")
        y = keras.layers.Conv2D(128, (7, 1), activation="relu")(inputs_seq)
        y = keras.layers.MaxPooling2D((7, 1))(y)
        y = keras.layers.Conv2D(128, (5, 1), activation="relu")(y)
        y = keras.layers.MaxPooling2D((5, 1))(y)
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Dense(512, activation="relu")(y)
        out_seq = keras.layers.Dense(6, activation="sigmoid", name="output")(y)

        self.submodel_seq = keras.Model(inputs=inputs_seq, outputs=out_seq, name="sequence_model")

    def call(self, inputs):
        """
        Expect inputs as a tuple: (image_input_tensor, sequence_input_tensor).
        Return dict of outputs from both submodels.
        """
        image_input, seq_input = inputs

        out_img = self.submodel_img(image_input)
        out_seq = self.submodel_seq(seq_input)

        return {"image_output": out_img, "sequence_output": out_seq}


def my_model_function():
    # Return an instance of MyModel, weights are randomly initialized by default.
    # In practice you'd load pretrained weights as needed.
    return MyModel()


def GetInput():
    """
    Return a tuple of inputs matching what MyModel expects:
    - image input tensor: shape (1, 224, 224, 3), float32
    - sequence input tensor: shape (1, 128, 9, 1), float32

    Inputs are random uniform tensors mimicking real inputs.
    """

    image_input = tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)
    seq_input = tf.random.uniform((1, 128, 9, 1), dtype=tf.float32)

    return (image_input, seq_input)

