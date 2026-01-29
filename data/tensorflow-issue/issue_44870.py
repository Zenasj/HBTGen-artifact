# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32) ← inferred input shape based on model summary and Input layers

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, concatenate
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam

# Placeholder loss function since original hn_multilabel_loss is referenced but not defined
def hn_multilabel_loss(y_true, y_pred):
    # This is a placeholder. Replace with the real implementation.
    return tf.reduce_mean(tf.square(y_true - y_pred))

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=13, input_shape=(128,128,1), **kwargs):
        super().__init__(**kwargs)
        # Instantiate 3 MobileNet models as submodules

        # Base MobileNet without top, weights=None as per original code
        self.input_shape_ = input_shape
        self.num_classes = num_classes

        # Create 3 MobileNet submodels with identical architecture but different names
        # Weights are None here as original models were trained and saved separately; here, we'll build new ones.

        self.mobilenet1 = self._build_mobilenet_submodel(name="model_1")
        self.mobilenet2 = self._build_mobilenet_submodel(name="model_2")
        self.mobilenet3 = self._build_mobilenet_submodel(name="model_3")

        # Final dense layer after concatenation (39 units from 3x13)
        self.merge_dense = Dense(num_classes, activation='sigmoid', name="output_layer")

    def _build_mobilenet_submodel(self, name):
        # Build a MobileNet-based submodel without top
        # Note: Since weights=None in original snippet, do not load pretrained
        base_model = MobileNet(input_shape=self.input_shape_, include_top=False, weights=None)
        # We'll manually build a Sequential-like model within subclass model
        # Layers per original build: base MobileNet -> GlobalAveragePooling2D -> Dropout(0.5) -> Dense(num_classes, sigmoid)
        global_avg = GlobalAveragePooling2D()
        dropout = Dropout(0.5)
        dense = Dense(self.num_classes, activation='sigmoid')

        # Save layers as attributes so they can be used in call
        return {
            "base": base_model,
            "gap": global_avg,
            "dropout": dropout,
            "dense": dense,
            "name": name
        }

    def call(self, inputs, training=False):
        # Forward pass through each submodel
        def submodel_forward(submodel_dict, x):
            x = submodel_dict["base"](x, training=training)
            x = submodel_dict["gap"](x)
            x = submodel_dict["dropout"](x, training=training)
            x = submodel_dict["dense"](x)
            return x

        out1 = submodel_forward(self.mobilenet1, inputs)
        out2 = submodel_forward(self.mobilenet2, inputs)
        out3 = submodel_forward(self.mobilenet3, inputs)

        # Concatenate outputs along feature axis
        merged = tf.concat([out1, out2, out3], axis=-1)

        # Final dense layer to output num_classes predictions with sigmoid activation
        output = self.merge_dense(merged)

        return output

def my_model_function():
    # Return an instance of MyModel with defaults matching the observed model
    # num_classes=13 from all_labels length inferred in original code
    # input_shape=(128,128,1) from model summary input_layer_main shape
    model = MyModel(num_classes=13, input_shape=(128,128,1))
    # Compile model with same optimizer, loss, and metrics as original ensemble
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    METRICS = [
        "binary_accuracy",
        "top_k_categorical_accuracy",
        hn_multilabel_loss,
        tf.keras.metrics.AUC(),
        'mae'
    ]
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)
    return model

def GetInput():
    # Generate a random tensor matching the input shape (batch size any, height=128, width=128, channels=1)
    # dtype float32 to match MobileNet expectation for input
    batch_size = 4  # Example batch size; can be dynamic
    return tf.random.uniform(shape=(batch_size, 128, 128, 1), dtype=tf.float32)

