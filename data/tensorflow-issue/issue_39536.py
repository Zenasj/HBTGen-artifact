# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from DenseNet121 input layers in ensemble

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self, n_models=5):
        super().__init__()
        self.n_models = n_models
        
        # Create n_models individual DenseNet121 based subnetworks
        self.submodels = []
        for i in range(n_models):
            inputs = Input(shape=(224, 224, 3), name=f'input_{i+1}')
            base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
            
            # Take intermediate layer output "pool3_conv" (valid in DenseNet121)
            output = base_model.get_layer("pool3_conv").output
            
            # Conv layers + BatchNorm as in original code
            x = Conv2D(128, 3, activation='relu', padding='same')(output)
            x = BatchNormalization()(x)
            x = Conv2D(64, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(2, activation='softmax', name=f'clf_output_{i+1}')(x)
            
            submodel = Model(inputs=inputs, outputs=x, name=f'submodel_{i+1}')
            
            # Freeze layers except input layer for ensemble usage (can be toggled later)
            for layer in submodel.layers[1:]:
                layer.trainable = False
                # Renaming layers to avoid name collisions
                layer._name = f'ensemble_{i+1}_{layer._name}'
            
            self.submodels.append(submodel)
        
        # Final ensemble layers
        # Inputs list: one input tensor per submodel
        self.stack_inputs = [sm.input for sm in self.submodels]
        # IMPORTANT: Instead of using submodel.output, use submodel(submodel.input) call 
        # to avoid dataset feeding issues (per issue resolution)
        self.stack_outputs = [sm(sm.input) for sm in self.submodels]
        
        self.concat = Concatenate()
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(2, activation='softmax')
        
    def call(self, inputs, training=False):
        # Inputs is expected as a list of tensors matching self.stack_inputs in order
        # Apply each submodel on its corresponding input
        outputs = []
        for i in range(self.n_models):
            # inputs[i] corresponds to input_i+1
            out = self.submodels[i](inputs[i], training=training)
            outputs.append(out)
        
        merged = self.concat(outputs)
        x = self.dense1(merged)
        x = self.dense2(x)
        
        return x

def my_model_function():
    """
    Returns an instance of MyModel, default 5 subnetworks.
    """
    model = MyModel(n_models=5)
    return model

def GetInput():
    """
    Returns a tuple of 5 random input tensors matching the expected inputs
    for MyModel, each of shape (batch_size, 224, 224, 3), dtype float32.
    Batch size is arbitrarily set to 16 matching the example.
    """
    batch_size = 16
    return tuple(
        tf.random.uniform(shape=(batch_size, 224, 224, 3), dtype=tf.float32)
        for _ in range(5)
    )

