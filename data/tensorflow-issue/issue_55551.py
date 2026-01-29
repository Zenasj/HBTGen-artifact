# tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32) ← Input shape inferred from original model usage

import tensorflow as tf

IMG_SIZE = 32
NUM_CLASSES = 10
NUM_FEATURES = 1 * 1 * 1280  # flattened feature size from MobileNetV2 base output
BATCH_SIZE = 32  # batch size used in training and inference


class MyModel(tf.keras.Model):
    def __init__(self):
        """
        This model fuses the base MobileNetV2 (feature extractor) with a small
        head dense model for transfer learning classification.

        The base_model outputs feature maps of shape (None, 1, 1, 1280).
        These are reshaped to (None, 1280) before feeding into the head_model.

        This matches the functionality of the original TransferLearningModel tf.Module,
        adapted into a single Keras.Model subclass for better composability and
        easier TFLite conversion.

        Assumptions:
        - Input images are float32 normalized to [0,1].
        - Head model uses two dense layers with relu and linear outputs.
        - Output logits are raw, so softmax needs to be applied externally or in infer method.
        """
        super().__init__()
        # Load MobileNetV2 with imagenet weights, exclude top layers, fixed input shape
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        # Freeze base model
        self.base_model.trainable = False

        # Head model: two dense layers as classifier
        self.head_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
            tf.keras.layers.Dense(NUM_CLASSES)  # logits output
        ])

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def call(self, inputs):
        """
        Forward pass for inference or training.

        Parameters:
            inputs: Input images tensor, shape (batch, 32, 32, 3), float32, normalized [0,1].

        Returns:
            Dictionary with key 'output' and softmax probabilities as the value,
            shape (batch, NUM_CLASSES).
        """
        # MobileNetV2 preprocessing expects images in [-1,1].
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
        features = self.base_model(x, training=False)  # (batch,1,1,1280)
        bottleneck = tf.reshape(features, (-1, NUM_FEATURES))  # flatten (batch,1280)
        logits = self.head_model(bottleneck)  # (batch, NUM_CLASSES)
        probs = tf.nn.softmax(logits)
        return {'output': probs}

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_FEATURES], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32)
    ])
    def train_step(self, bottleneck, label):
        """
        Performs one training step using bottleneck features.

        Parameters:
            bottleneck: Tensor of shape (batch, 1280), float32 bottleneck features.
            label: One-hot labels tensor of shape (batch, 10), float32.

        Returns:
            Dictionary with "loss" and gradients keyed by variable name.
        """
        with tf.GradientTape() as tape:
            logits = self.head_model(bottleneck)
            prediction = tf.nn.softmax(logits)
            # Use predefined categorical crossentropy loss
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            loss = loss_fn(label, logits)
        gradients = tape.gradient(loss, self.head_model.trainable_variables)
        # Use SGD optimizer with fixed learning rate like original
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        optimizer.apply_gradients(zip(gradients, self.head_model.trainable_variables))

        result = {"loss": loss}
        for var, grad in zip(self.head_model.trainable_variables, gradients):
            if grad is not None:
                result[var.name] = grad
        return result

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save_head_weights(self, checkpoint_path: str):
        """
        Save head model weights to a checkpoint file.

        Parameters:
            checkpoint_path: string path filename.

        Returns:
            Dict with checkpoint_path for confirmation.
        """
        tensor_names = [w.name for w in self.head_model.weights]
        tensors_to_save = [w.read_value() for w in self.head_model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save')
        return {'checkpoint_path': checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore_head_weights(self, checkpoint_path: str):
        """
        Restore head model weights from a checkpoint file.

        Parameters:
            checkpoint_path: string path filename.

        Returns:
            Dictionary of restored tensors keyed by their variable names.
        """
        restored_tensors = {}
        for tensor in self.head_model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path,
                tensor_name=tensor.name,
                dt=tensor.dtype,
                name='restore')
            tensor.assign(restored)
            restored_tensors[tensor.name] = restored
        return restored_tensors

    @tf.function
    def extract_head_weights(self):
        """
        Extract trainable weights of the head model.

        Returns:
            Dictionary mapping variable names to their values.
        """
        weights_dict = {}
        for weight in self.head_model.weights:
            weights_dict[weight.name] = weight.read_value()
        return weights_dict


def my_model_function():
    # Returns an instance of MyModel with MobileNetV2 base + head
    return MyModel()


def GetInput():
    # Return a batch of random images with shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3),
    # dtype float32 normalized in [0,1]
    return tf.random.uniform(
        (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

