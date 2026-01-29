# tf.random.uniform((H, W, 3), dtype=tf.float32)
import tensorflow as tf

# Based on the DeepDream tutorial for TensorFlow 2.0:
# Input images are expected to be float tensors with 3 channels (RGB),
# with shape (height, width, 3), batch dimension is not explicitly used here,
# so we assume a single image tensor input.

def calc_loss(img, model):
    # Placeholder loss calculation function inferred from tutorial context.
    # Typically, you select certain layers from the model and compute activations norm.
    # Here we assume model outputs a dictionary of activations with layer names as keys.
    # We sum the mean activations of some layers to generate a loss to maximize.
    # Placeholder uses "mixed10" layer output as in InceptionV3 deepdream tutorial.
    activations = model(img)
    # Assuming model(img) returns a dict with layer outputs.
    # If not, adapt as needed.
    loss = tf.zeros(shape=())
    # Using "mixed10" layer if present
    if "mixed10" in activations:
        loss += tf.reduce_mean(activations["mixed10"])
    else:
        # Fallback: sum all activations means if a dict with tensors is returned
        if isinstance(activations, dict):
            for act in activations.values():
                loss += tf.reduce_mean(act)
        else:
            loss = tf.reduce_mean(activations)
    return loss


class MyModel(tf.keras.Model):
    # Fusing the core idea from the reported DeepDream class:
    # handle input image, step count, and step size.
    # We must NOT iterate over tf.Tensors explicitly to avoid the OperatorNotAllowedInGraphError.
    #
    # Solution inferred: Replace Python for-loop over tf.range(...) tensor
    # with TensorFlow control flow (tf.while_loop) or tf.function with autograph-compatible iteration.
    # Since tf.function was causing the error in original code due to Python looping over tf.Tensor,
    # we will fix this by using tf.while_loop to maintain autograph compatibility and not cause errors.
    #
    # Also ensure input usage is compatible with tf.function jit_compile=True.
    
    def __init__(self, model=None):
        super().__init__()
        # Accepts a tf.keras Model that outputs intermediate activations as a dict or tensor.
        # For inference, we expect a pre-built model prepared for deep dream.
        self.model = model
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),  # Image tensor (H, W, 3)
        tf.TensorSpec(shape=(), dtype=tf.int32),                # steps scalar
        tf.TensorSpec(shape=(), dtype=tf.float32)               # step_size scalar
    ])
    def call(self, img, steps, step_size):
        tf.print("Tracing MyModel.call")
        
        # We must use a tf.Variable for img to track gradients and update it.
        img_var = tf.Variable(img)
        
        # Loop variables for tf.while_loop
        def cond(n, img_var, loss):
            return n < steps

        def body(n, img_var, loss):
            with tf.GradientTape() as tape:
                tape.watch(img_var)
                # Forward pass to get loss to maximize
                curr_loss = calc_loss(img_var, self.model)
            gradients = tape.gradient(curr_loss, img_var)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img_var.assign_add(gradients * step_size)
            img_var.assign(tf.clip_by_value(img_var, -1.0, 1.0))
            return n + 1, img_var, curr_loss

        # Initialize loss as zero scalar to satisfy loop variables
        n0 = tf.constant(0)
        loss0 = tf.constant(0.0)

        # Run the tf.while_loop for gradient ascent steps
        n_final, img_final, loss_final = tf.while_loop(cond, body, loop_vars=[n0, img_var, loss0])

        return loss_final, img_final


def my_model_function():
    # Instantiate MyModel with an InceptionV3 feature extraction model prepared for deepdream.
    # The original DeepDream tutorial uses a pretrained InceptionV3 model,
    # outputs intermediate layer activations as a dict.
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    
    # Select layers for feature extraction (per tutorial)
    layer_names = [
        'mixed3',   # low level features
        'mixed5',   # mid level features
        'mixed7',   # higher level features
        'mixed10',  # highest level features used for deep dream
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs={name: layer for name, layer in zip(layer_names, layers)})
    
    return MyModel(model=dream_model)


def GetInput():
    # Generate a random image tensor matching input shape (H, W, 3), floating point type
    # DeepDream tutorial uses images scaled roughly in [-1, 1]
    # We'll generate a 224x224 RGB image as a reasonable default
    H, W = 224, 224
    # Uniform random in [-1,1]
    img = tf.random.uniform((H, W, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return img

