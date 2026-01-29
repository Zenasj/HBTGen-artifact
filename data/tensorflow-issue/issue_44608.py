# tf.random.uniform((B, 299, 299, 3), dtype=tf.float32)  # Input shape inferred from issue code: 299x299 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.img_size = (299, 299)
        self.input_shape_ = (*self.img_size, 3)

        # Data augmentation layers
        data_augmentation_layers = [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode="constant"),
        ]
        self.data_augmentation = tf.keras.Sequential(data_augmentation_layers)

        # Preprocessing function from InceptionResNetV2
        # Using a Lambda layer here to apply tf.keras.applications.inception_resnet_v2.preprocess_input
        self.preprocess_input = tf.keras.layers.Lambda(
            tf.keras.applications.inception_resnet_v2.preprocess_input,
            name="preprocessing"
        )

        # Base model without top, weights initialized from imagenet, not trainable
        self.base_model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.input_shape_,
            include_top=False,
            weights='imagenet',
        )
        self.base_model.trainable = False

        # Global average pooling after base model
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Top classifier layers (dense + dropout + final dense sigmoid output)
        self.top_dense_1 = tf.keras.layers.Dense(
            512,
            activation="relu",
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_uniform"
        )
        self.top_dropout_1 = tf.keras.layers.Dropout(0.2)
        self.prediction_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_uniform"
        )
        # Additional dropout before top dense layers as per snippet
        self.dropout_base_output = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: float tensor with shape (batch_size, 299, 299, 3)
        """
        x = self.data_augmentation(inputs, training=training)
        x = self.preprocess_input(x)  # scale inputs like InceptionResNetV2 expects
        x = self.base_model(x, training=False)  # keep base_model in inference mode to avoid updating BatchNorm
        x = self.global_avg_pool(x)
        x = self.dropout_base_output(x, training=training)
        x = self.top_dense_1(x)
        x = self.top_dropout_1(x, training=training)
        outputs = self.prediction_layer(x)
        return outputs

    def get_intermediate_base_output(self, inputs):
        """
        Return the intermediate output of the base model (InceptionResNetV2 output)
        given the same inputs.
        This replicates usage of:
            model2 = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer('inception_resnet_v2').get_output_at(0))
        => Which fetches the connected output node from the base model layer.
        """
        x = self.data_augmentation(inputs, training=False)
        x = self.preprocess_input(x)
        # We must explicitly get the output from base_model with get_output_at(0) logic,
        # but since base_model is a Functional model, calling it directly gives output at 0th node.
        # Thus here direct call suffices.
        base_output = self.base_model(x, training=False)
        return base_output


def my_model_function():
    """
    Constructs and returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a random tensor input that matches the expected input shape of MyModel:
    Batch size of 1, shape (299, 299, 3), dtype float32.
    Values in [0, 255) typical for image input (before preprocessing).
    """
    return tf.random.uniform((1, 299, 299, 3), minval=0, maxval=255, dtype=tf.float32)

