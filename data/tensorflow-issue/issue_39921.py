# tf.random.uniform((B, 5, 1, 14), dtype=tf.float32), tf.random.uniform((B, 512, 6, 18), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Reshape
from tensorflow.keras import Model, Input

class MyConcat(tf.keras.layers.Layer):
    def __init__(self):
        super(MyConcat, self).__init__()

    def call(self, inputs):
        x, emb = inputs
        # Concatenate along channels axis (axis=1 for channels_first)
        return Concatenate(axis=1)([x, emb])

    def compute_output_shape(self, input_shape):
        # shape = (batch, channels_x + channels_emb, height, width)
        shape = (None,
                 input_shape[0][1] + input_shape[1][1],
                 input_shape[0][2],
                 input_shape[0][3])
        return shape


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Conv2DTranspose with data_format channels_first
        self.upconv1 = Conv2DTranspose(
            filters=16,
            kernel_size=(6, 5),
            strides=(6, 1),
            data_format="channels_first"
        )

        # Reshape layer to fix shape inference issue before concatenation
        # The user reported that adding a Reshape fixes saving issue.
        # Based on input shape (5,1,14), after Conv2DTranspose:
        # output channels=16, height and width need to be fixed to (6,14)
        # After strides (6,1) and kernel (6,5):
        # Output height: (5-1)*6 + 6 = 30; output width: (1-1)*1 + 5 =5
        # But user wants compatible shape to emb with height=6, width=18 for concatenation axis=1
        # Since spatial dimensions don't match, we cannot concatenate directly.
        # So, we will reshape output to (16, 6, 18) to match emb for concat.
        # This is a reasonable assumption based on user's comment on adding reshape.
        self.reshape = Reshape(target_shape=(16, 6, 18))

        self.concat = MyConcat()

    def call(self, inputs):
        x, emb = inputs  # expecting tuple/list of two tensors

        x = self.upconv1(x)
        x = self.reshape(x)
        out = self.concat([x, emb])
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tuple of random inputs that fit the model input shapes
    # inputs: (batch, 5, 1, 14), emb: (batch, 512, 6, 18)
    # Using batch size = 4 as example
    batch_size = 4
    input_tensor = tf.random.uniform(shape=(batch_size, 5, 1, 14), dtype=tf.float32)
    emb_tensor = tf.random.uniform(shape=(batch_size, 512, 6, 18), dtype=tf.float32)
    return (input_tensor, emb_tensor)

