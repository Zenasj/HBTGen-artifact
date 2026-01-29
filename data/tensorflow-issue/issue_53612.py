# tf.random.uniform((B, IMAGE_SIZE[0], IMAGE_SIZE[1], IMG_CHANNELS), dtype=tf.float32)
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

# For this example, let's assume some constants from the issue context:
IMAGE_SIZE = (224, 224)  # typical MobileNetV2 input size
IMG_CHANNELS = 3
CLASS_NAMES = ["class1", "class2", "class3"]  # placeholder classes
N_EPOCHS = 5  # example epoch count

# Placeholder quantization function from the issue context (likely they used this)
def apply_quantization_to_dense(layer):
    # Quantize only Dense layers for example purpose; pass-through otherwise
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

# Assumptions:
# - The model input shape is (B, 224, 224, 3) float32
# - The model includes preprocessing integrated in the model (using tf.keras.layers.Lambda)
# - The model replicates the MobileNetV2 + custom heads architecture shown in the code chunks
# - The original error relates to Lambda layer usage breaking multi-GPU
# - We'll replace Lambda preprocessing with a custom subclass layer to ensure serializability



class PreprocessingLayer(tf.keras.layers.Layer):
    """Custom preprocessing layer to replace Lambda for better multi-GPU compatibility"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Cast to float32 and preprocess input for MobileNetV2
        x = tf.cast(inputs, tf.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)  # note mobilenet_v2 module

    def get_config(self):
        # To ensure layer is serializable for multi-GPU usage
        base_config = super().get_config()
        return base_config

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Preprocessing replaced from Lambda to custom Layer for better serialization
        self.preprocessing = PreprocessingLayer(input_shape=[*IMAGE_SIZE, IMG_CHANNELS])
        # Load base MobileNetV2 without top, with imagenet weights
        self.base_model_pretrained = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=[*IMAGE_SIZE, IMG_CHANNELS]
        )
        self.base_model_pretrained.trainable = True  # fine tuning

        # Clone model to apply quantization to Dense layers (simulating original logic)
        # Since apply_quantization_to_dense is a placeholder, we replicate quantization with annotate then apply outside
        def quantize_clone(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            return layer

        self.q_aware_pretrained_model = tf.keras.models.clone_model(
            self.base_model_pretrained,
            clone_function=quantize_clone
        )
        self.q_aware_pretrained_model._name = 'custom_mnet_trainable'

        # Sequential base model combining preprocessing, quant aware pretrained, global pool
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(64, name='object_dense',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False, center=False)
        self.act1 = tf.keras.layers.Activation('relu', name='relu_dense_64')
        self.drop1 = tf.keras.layers.Dropout(rate=0.5, name='dropout_dense_64')

        self.dense2 = tf.keras.layers.Dense(32, name='object_dense_2',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False, center=False)
        self.act2 = tf.keras.layers.Activation('relu', name='relu_dense_32')
        self.drop2 = tf.keras.layers.Dropout(rate=0.4, name='dropout_dense_32')

        self.dense3 = tf.keras.layers.Dense(16, name='object_dense_16',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.out_layer = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name='object_prob')

    def call(self, inputs, training=False):
        # Forward pass integrating preprocessing and the base + heads
        x = self.preprocessing(inputs)
        x = self.q_aware_pretrained_model(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        x = self.dense3(x)
        out = self.out_layer(x)
        return out


def my_model_function():
    # Returns an instance of MyModel, already initialized
    model = MyModel()
    # According to the issue, they apply quantize_apply after model cloning
    # So we apply quantize_apply here
    quantized_model = tfmot.quantization.keras.quantize_apply(model)
    return quantized_model


def GetInput():
    # Return a random tensor input matching the model input shape (B, H, W, C)
    # Batch size is arbitrary, e.g., 8 for demonstration
    BATCH_SIZE = 8
    return tf.random.uniform((BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMG_CHANNELS), dtype=tf.float32)

