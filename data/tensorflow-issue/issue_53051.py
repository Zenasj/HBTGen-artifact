# tf.random.uniform((B, H, W, 3), dtype=tf.float32) ← The model expects input images with 3 channels

import tensorflow as tf

_default_channel_n = 3  # RGB images assumed
CURRENT_DEVICE = 'cpu'  # Default device, can be 'tpu' to simulate TPU context

def auto_tpu(device='cpu'):
    """Automatically open context manager for TPU or CPU/GPU."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            import time
            s = time.time()
            if device == 'tpu':
                # Placeholder for actual TPU strategy scope, e.g.:
                # with tf.distribute.TPUStrategy(...).scope():
                # Simulated as no real TPU available here
                ret = fn(*args, **kwargs)
            else:
                ret = fn(*args, **kwargs)
            e = time.time()
            print(f'device: {repr(device)}, time elapsed: {e-s:.3f} second(s)')
            return ret
        return wrapper
    return decorator

def create_augmentation_model(image_input_hw, mask_input_hw, class_n: int):
    """
    Creates an augmentation model that applies runtime preprocessing.
    Note: This augmentation currently causes TPU compile-time constant errors
    due to random ops in Keras preprocessing layers.
    """
    seq = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
        ],
        name='sequential_augmentation_layers'
    )

    image_input_shape = list(image_input_hw) + [_default_channel_n]
    mask_input_shape = list(mask_input_hw) + [class_n]

    x_im = tf.keras.Input(shape=image_input_shape, name='image_input')
    x_ma = tf.keras.Input(shape=mask_input_shape, name='mask_input')

    # Apply augmentation sequence independently to both inputs
    # Note: Applying the same augmentation pipeline to image and mask is simplistic
    # and likely incorrect for masks since their semantics differ.
    aug_im = seq(x_im)
    aug_ma = seq(x_ma)  # Mask augmentation with image augmentation layers may require custom layers
    
    return tf.keras.Model(
        inputs=[x_im, x_ma],
        outputs=[aug_im, aug_ma],
        name='sequential_augmentation_model'
    )

def create_segmentation_model(class_n: int):
    """
    Dummy segmentation model for demonstration.
    Takes input tensor of shape (H, W, 3) and outputs segmentation mask of shape (H, W, class_n).
    """
    inputs = tf.keras.Input(shape=(None, None, _default_channel_n), name='segmentation_input')
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(class_n, 1, padding="same", activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name='segmentation_model')

class MyModel(tf.keras.Model):
    """
    Fused model combining augmentation and segmentation.
    In training, input images and masks are augmented via augmentation_model.
    Then segmentation_model is run on augmented images.
    The output is the predicted segmentation mask.

    This model replicates the behavior of AugConcatedSegModel as described.
    """
    def __init__(self,
                 image_input_hw,
                 mask_input_hw,
                 class_n,
                 augmentation_model,
                 segmentation_model,
                 **kwargs):
        # Input shape for images: (H, W, 3)
        x_im = tf.keras.Input(shape=list(image_input_hw) + [_default_channel_n], name="image_input")
        # Input shape for masks: (H, W, class_n)
        x_ma = tf.keras.Input(shape=list(mask_input_hw) + [class_n], name="mask_input")

        # Store models for use later
        self.augmentation_model = augmentation_model
        self.segmentation_model = segmentation_model

        # Initialize base tf.keras.Model with image input only to match usage later 
        # Note: Due to augmentation applied to both inputs, forward signature needs a tuple
        # We'll handle inputs as a tuple (image_tensor, mask_tensor) in call
        super().__init__(inputs=[x_im, x_ma], outputs=self.call([x_im, x_ma]), **kwargs)

    def call(self, inputs, training=False):
        # inputs expected as tuple/list: (image_tensor, mask_tensor)
        im, ma = inputs

        if training:
            # Apply augmentation model during training to both inputs
            im_aug, ma_aug = self.augmentation_model([im, ma])
        else:
            # No augmentation during inference
            im_aug, ma_aug = im, ma

        # Run segmentation model on augmented images
        seg_pred = self.segmentation_model(im_aug, training=training)
        # Output predicted segmentation mask
        return seg_pred

    def train_step(self, data):
        # data is expected as tuple (image_batch, mask_batch)
        im, ma = data
        # In train_step we want to apply the augmentation_model manually to inputs
        im_aug, ma_aug = self.augmentation_model([im, ma])

        with tf.GradientTape() as tape:
            # Forward pass on augmented images
            ma_pred = self.segmentation_model(im_aug, training=True)

            # Compute the loss value using the compiled loss
            loss = self.compiled_loss(ma_aug, ma_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the loss metric)
        self.compiled_metrics.update_state(ma_aug, ma_pred)
        # Return metric results as a dict
        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    # Using an example input spatial shape and classes
    image_input_hw = (128, 128)  # height, width of input images
    mask_input_hw = (128, 128)   # height, width of segmentation mask
    class_n = 4                  # number of segmentation classes

    # Create augmentation model
    aug_model = create_augmentation_model(image_input_hw, mask_input_hw, class_n)

    # Create segmentation model
    seg_model = create_segmentation_model(class_n)

    # Create fused model with augmentation and segmentation
    model = MyModel(image_input_hw, mask_input_hw, class_n,
                    augmentation_model=aug_model,
                    segmentation_model=seg_model,
                    name="my_augmented_segmentation_model")
    return model

def GetInput():
    """
    Generate a random input tuple (image_batch, mask_batch) compatible with MyModel.
    Here batch size is set arbitrarily as 2 (could be any reasonable number).
    Image pixel values are floats in [0,1], mask is one-hot encoded masks.
    """
    batch_size = 2
    height = 128
    width = 128
    class_n = 4

    # Random float images (batch, H, W, 3)
    images = tf.random.uniform((batch_size, height, width, _default_channel_n), dtype=tf.float32)

    # Random integer mask labels (batch, H, W), values in [0, class_n-1]
    labels_int = tf.random.uniform((batch_size, height, width), maxval=class_n, dtype=tf.int32)

    # One-hot encode masks (batch, H, W, class_n)
    masks = tf.one_hot(labels_int, depth=class_n, axis=-1, dtype=tf.float32)

    return (images, masks)

