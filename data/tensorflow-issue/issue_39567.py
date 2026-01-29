# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← input shape based on original model input

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self, num_models=2, input_shape=(224, 224, 3), num_classes=2):
        """
        A fused ensembled model consisting of multiple DenseNet121-based submodels.
        Each submodel outputs a 2-class softmax prediction via custom convolutional layers.
        The outputs of all submodels are concatenated, followed by two dense layers
        to produce the final ensemble prediction.

        Args:
          num_models: Number of base submodels to ensemble.
          input_shape: Shape of a single input image (H, W, C).
          num_classes: Number of classes for classification output.
        """
        super().__init__()

        self.num_models = num_models
        self.input_shape_ = input_shape
        self.num_classes = num_classes

        # Create Input layer
        self.inputs_ = Input(shape=input_shape, name='ensemble_input')

        # Build base submodels and rename their layers to avoid name conflicts
        self.submodels = []
        for i in range(num_models):
            # Build individual base model for each submodel using the functional API
            # with its own copy of DenseNet121 base without top
            base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
            # Extract output from specific layer
            base_output = base_model.get_layer("pool3_conv").output
            x = Conv2D(128, 3, activation='relu', padding='same')(base_output)
            x = BatchNormalization()(x)
            x = Conv2D(64, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(num_classes, activation='softmax', name='clf_output')(x)
            model = Model(inputs=base_model.input, outputs=x, name=f'base_model_{i}')
            
            # Rename layers in submodel to avoid same layer names between submodels for saving sanity
            for layer in model.layers[1:]:  # typically skip InputLayer
                old_name = layer.name
                layer._name = f'ensemble_{i+1}_{old_name}'

            self.submodels.append(model)

        # Since we use Functional models for submodels, we treat the inputs as single input tensor

        # We will execute all submodels by passing the same input tensor and collect outputs
        # The outputs are concatenated together
        self.concat = Concatenate(name='ensemble_concat')

        self.dense1 = Dense(16, activation='relu', name='ensemble_dense_1')
        self.dense2 = Dense(num_classes, activation='softmax', name='ensemble_dense_2')

    def call(self, inputs, training=False):
        """
        Forward pass of the ensembled model.
        Args:
            inputs: Tensor of shape (batch_size, 224, 224, 3)
            training: Bool, whether in training mode or not
        Returns:
            Tensor of shape (batch_size, num_classes) with ensemble softmax prediction
        """
        outputs = []
        for model in self.submodels:
            # Call each submodel on the same input tensor
            out = model(inputs, training=training)
            outputs.append(out)
        x = self.concat(outputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    # Assumptions: 2 base submodels, input 224x224x3, 2-class classification
    return MyModel(num_models=2)

def GetInput():
    # Return a random tensor input compatible with the model: batch size 1, shape (1, 224, 224, 3)
    # dtype float32, values in [0, 1)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

