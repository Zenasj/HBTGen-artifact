# tf.random.uniform((1, 20), dtype=tf.float32) ← inferred input shape and dtype from training data

import tensorflow as tf
import tensorflow_model_optimization as tfmot

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Baseline model: simple sequential with Dense(20) + Flatten
        # The input shape is (20,)
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, input_shape=(20,)),
            tf.keras.layers.Flatten()
        ])
        # Wrap the baseline model with pruning wrapper for pruned model
        self.prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        self.pruned_model = self.prune_low_magnitude(self.base_model)
        
        # For comparison, we replicate the "stripped" pruned model,
        # but we do that in the call function by stripping pruning from pruned_model.
        # The stripped model corresponds to model_for_export in original code.

    def call(self, inputs, training=False):
        # Run baseline model
        base_out = self.base_model(inputs, training=training)
        # Run pruned model (with pruning wrappers active if training)
        pruned_out = self.pruned_model(inputs, training=training)
        # Strip pruning wrappers from pruned model for export comparison after call
        # Here we simulate stripping on the fly by calling tfmot strip_pruning method,
        # but it expects a model to strip, not a call output.
        # So, we replicate the logic and return the stripped model output as a call:

        # However, stripping pruning is typically done offline (not on the fly).
        # Here for demonstration, we just call baseline model again to simulate stripping,
        # since stripping returns a model with same weights but without pruning masks.
        # The stripped model should behave like baseline but internally have pruned weights.

        # In this context, to meet requirements, return a tuple with:
        # (baseline output, pruned output, stripped model output)

        # Note: stripped model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        # But can't do it inside call. We will simulate by using a stripped model attribute.

        # We rely on a pre-stripped model instance (assigned externally or inside a method)
        if hasattr(self, "stripped_model"):
            stripped_out = self.stripped_model(inputs, training=False)
        else:
            # fallback: just re-use base model output as placeholder for stripped output
            stripped_out = base_out

        return base_out, pruned_out, stripped_out

    def setup_stripped_model(self):
        # Create a stripped version of the pruned model to simulate export
        self.stripped_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)


def my_model_function():
    model = MyModel()
    # Setup stripped model (simulate export stripping pruning wrappers)
    model.setup_stripped_model()

    # Compile and "pretrain" weights on dummy data (mimicking original code)
    x_train = tf.random.normal((1, 20), dtype=tf.float32)
    # dummy one-hot labels with num_classes=20
    y_train = tf.one_hot([0], depth=20, dtype=tf.float32)

    model.base_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=['accuracy']
    )
    # Train the baseline model for one step to initialize weights roughly like original
    model.base_model.fit(x_train, y_train, epochs=1, verbose=0)
    # Load the trained weights into pruned_model wrappers by calling prune_low_magnitude again
    # The pruned model weights are linked to baseline model weights inside wrapper,
    # so they share weights. No extra loading needed here.

    # Compile pruned_model as well (optional, pruning requires compile for finetuning)
    model.pruned_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def GetInput():
    # Return a random input tensor with shape (1, 20), dtype float32, to match model input
    return tf.random.uniform((1, 20), dtype=tf.float32)

