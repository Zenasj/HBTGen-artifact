# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape (batch size, 1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer applied to main input
        self.dense = tf.keras.layers.Dense(1)
        # Note: loss_input is not connected to any layer, but used in loss function

    def call(self, inputs):
        # inputs is a tuple/list of two tensors: (main_input, loss_input)
        main_input, loss_input = inputs
        output = self.dense(main_input)
        # Return output, and the loss_input so it can be accessed externally for loss
        # (This mimics the original setup where loss_input is input but not connected in model graph)
        return output, loss_input

def custom_loss_fn(loss_input):
    def loss(y_true, y_pred):
        # The custom loss multiplies y_true * y_pred * loss_input
        # loss_input has to be broadcast-compatible with y_true and y_pred
        return y_true * y_pred * loss_input
    return loss

def my_model_function():
    # Create an instance of MyModel
    return MyModel()

def GetInput():
    # Generate example input compatible with MyModel
    # Here we generate a batch of 4 samples with shape (1,) for each input
    batch_size = 4
    main_input = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    loss_input = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    return (main_input, loss_input)

# Note:
# The original TF.keras Model usage had a problem saving/loading properly because
# "loss_input" was not connected to any layer downstream and thus ignored during save.
# 
# This rewrite encapsulates this design in a single subclassed Model keeping loss_input 
# explicitly returned so it can be used in the custom loss function externally.
# 
# A user should compile the model similarly using `custom_loss_fn(loss_input)` 
# provided during training, handling loss_input as external input tensor.
#
# This code is compatible with TF 2.20.0, and can be compiled with XLA as:
#
# model = my_model_function()
# @tf.function(jit_compile=True)
# def compiled(inputs):
#     return model(inputs)
#
# This design acknowledges the limitation that input tensors not connected in the graph
# are tricky to serialize or save in Keras's Functional API. Using subclassed Model 
# and explicitly passing inputs in call makes this clearer and more manageable.

