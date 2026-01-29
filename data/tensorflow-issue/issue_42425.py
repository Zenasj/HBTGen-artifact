# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape (batch size, height, width, channels) typical for object detection models like ResNet50 backbone

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load the SavedModel via tf.saved_model.load
        # Assuming the model is an object detection model with a signature 'serving_default'
        self.loaded = tf.saved_model.load('model')  # Assumes folder 'model' contains the SavedModel

        # Extract the callable signature for inference
        try:
            self.infer = self.loaded.signatures['serving_default']
        except Exception:
            # Fallback: use callable loaded object itself
            self.infer = self.loaded

        # Store variables if needed, though not used explicitly here
        self.variables_list = self.loaded.variables if hasattr(self.loaded, 'variables') else []

    def call(self, inputs):
        # The inputs should match the input signature of serving_default
        # It often expects a dict keyed by input tensor names, but here we assume single input tensor
        # Prepare input according to signature inputs:
        if isinstance(inputs, dict):
            # If inputs is already dict (pre-packaged), use directly
            infer_inputs = inputs
        else:
            # Attempt to get input key from signature inputs
            # serving_default.structured_input_signature returns a tuple: (args, kwargs)
            input_signature = self.infer.structured_input_signature
            # input_signature is like ((input_tensor_spec,), {})
            # Grab the first argument key (likely string key for dict)
            try:
                kwargs = input_signature[1]
                if kwargs:
                    input_key = list(kwargs.keys())[0]
                    infer_inputs = {input_key: inputs}
                else:
                    # No kwargs, just positional args - feed as is in a tuple
                    infer_inputs = (inputs,)
            except Exception:
                infer_inputs = (inputs,)

        # Call the loaded Serving function
        output = self.infer(**infer_inputs) if isinstance(infer_inputs, dict) else self.infer(*infer_inputs)

        # The output can be a dict of tensors
        # For a clean output, detect if it is a dict and return first tensor or whole dict
        if isinstance(output, dict):
            # Return outputs as a dict for full info (e.g., detection boxes, scores)
            return output
        else:
            return output

def my_model_function():
    """
    Returns an instance of MyModel.
    Assumes the SavedModel directory 'model' exists in working dir and compatible with TF 2.20.0.
    """
    return MyModel()

def GetInput():
    """
    Generates a random input tensor matching the typical input for object detection networks like ResNet50 backbone:
    Assumed shape: batch_size=1, height=224, width=224, channels=3, float32 dtype,
    as common for image feature backbones.

    NOTE: The exact expected input shape should match that from the SavedModel's input signature.
    This is an assumption based on typical image sizes and TF image models.
    """
    batch_size = 1
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

