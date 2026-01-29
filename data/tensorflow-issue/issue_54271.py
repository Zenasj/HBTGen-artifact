# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) ← inferred input shape per each of the 3 inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        IMG_SHAPE = (64, 64, 3)

        # Base pretrained models - MobileNetV3Large without top layers, pooling by max, dropout 0.2
        # Disabled training (training=False in call), load ImageNet weights
        self.base_model1 = tf.keras.applications.MobileNetV3Large(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet',
            minimalistic=False,
            pooling='max',
            dropout_rate=0.2)
        self.base_model2 = tf.keras.applications.MobileNetV3Large(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet',
            minimalistic=False,
            pooling='max',
            dropout_rate=0.2)
        self.base_model3 = tf.keras.applications.MobileNetV3Large(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet',
            minimalistic=False,
            pooling='max',
            dropout_rate=0.2)

        # Ensure the layer names won't clash (as per original code renaming – 
        # here just documenting as comment)
        # self.base_model1._name = "MobilenetV3large1"
        # self.base_model2._name = "MobilenetV3large2"
        # self.base_model3._name = "MobilenetV3large3"

        # Dense layers before concatenation with ReLU activation
        self.pre_concat_layer1 = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros')
        self.pre_concat_layer2 = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros')
        self.pre_concat_layer3 = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros')

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Post concatenation layers
        self.post_concat_layer = tf.keras.layers.Dense(
            128, activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros')
        self.prediction_layer = tf.keras.layers.Dense(
            2, activation='softmax',
            kernel_initializer='random_uniform',
            bias_initializer='zeros')

    def call(self, inputs, training=False):
        # inputs is expected to be a tuple or list of three tensors
        input1, input2, input3 = inputs

        # Process each input through respective base model and Dense layers
        # Base models are called with training=False to keep pretrained
        # batchnorm layers in inference mode.

        x1 = self.base_model1(input1, training=False)
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        x1 = self.dropout(x1, training=training)
        x1 = self.batch_norm(x1, training=training)
        x1 = self.pre_concat_layer1(x1)
        x1 = self.dropout(x1, training=training)
        x1 = self.batch_norm(x1, training=training)

        x2 = self.base_model2(input2, training=False)
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
        x2 = self.dropout(x2, training=training)
        x2 = self.batch_norm(x2, training=training)
        x2 = self.pre_concat_layer2(x2)
        x2 = self.dropout(x2, training=training)
        x2 = self.batch_norm(x2, training=training)

        x3 = self.base_model3(input3, training=False)
        x3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
        x3 = self.dropout(x3, training=training)
        x3 = self.batch_norm(x3, training=training)
        x3 = self.pre_concat_layer3(x3)
        x3 = self.dropout(x3, training=training)
        x3 = self.batch_norm(x3, training=training)

        # Concatenate outputs from the 3 branches
        concatenated = tf.keras.layers.concatenate([x1, x2, x3], axis=-1)

        # Post concatenation processing
        x = self.post_concat_layer(concatenated)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)

        # Final prediction layer: 2-class softmax
        output = self.prediction_layer(x)
        return output


def my_model_function():
    # Return a new instance of MyModel
    model = MyModel()

    # Since base MobileNetV3Large loads pretrained ImageNet weights automatically,
    # no further weight loading is necessary here.
    return model


def GetInput():
    # Return a tuple of three random input tensors corresponding to:
    # batch size 32 (typical batch size from original code)
    # input shape (64, 64, 3) per image branch
    # dtype float32 typical for images scaled 0..1 (random uniform)
    BATCH_SIZE = 32
    input_shape = (64, 64, 3)
    input1 = tf.random.uniform((BATCH_SIZE,) + input_shape, dtype=tf.float32)
    input2 = tf.random.uniform((BATCH_SIZE,) + input_shape, dtype=tf.float32)
    input3 = tf.random.uniform((BATCH_SIZE,) + input_shape, dtype=tf.float32)
    return (input1, input2, input3)

