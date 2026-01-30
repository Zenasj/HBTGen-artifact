import tensorflow as tf
from tensorflow import keras

3
def new_concatenated_model(
    image_input_hw,
    mask_input_hw,
    class_n
):
    seg_model = create_segmentation_model(class_n)
    aug_model = create_augmentation_model(
        image_input_hw, mask_input_hw, class_n)
    
    image_input_shape = list(image_input_hw) + [3]

    @auto_tpu(device=CURRENT_DEVICE) # decorator `auto_tpu` is just context manager.
    def create():
        im = seg_model.input
        model = AugConcatedSegModel(
            inputs=im,
            outputs=seg_model(im),
            augmentation_model=aug_model,
            name='seg_model_train_with_aug'
        )
        return model
    
    model = create()
    return model

3
class AugConcatedSegModel(tf.keras.Model):
    def __init__(
        self,
        inputs=None,
        outputs=None,
        augmentation_model=None, 
        **kwargs
    ):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.augmentation_model = augmentation_model

    def train_step(self, data):
        im, ma = data
        im, ma = self.augmentation_model((im, ma))

        with tf.GradientTape() as tape:
            ma_pred = self(im, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(ma, ma_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(ma, ma_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}