# tf.random.uniform((1, 32, 64, 64, 1), dtype=tf.float32) ← Assuming a 5D input typical for 3D conv 
import tensorflow as tf
from tensorflow.keras import layers

class C3BR(tf.keras.Model):
    def __init__(self, filterNum, kSize, strSize, padMode, dFormat='channels_first'):
        super(C3BR, self).__init__()
        if dFormat == 'channels_first':
            self.conAx = 1
            self.data_format = 'channels_first'
        else:
            self.conAx = -1
            self.data_format = 'channels_last'
        self.kSize = (kSize, kSize, kSize)
        # 3D conv layer with specified filters, kernel size, strides, padding, data format
        self.conv = layers.Conv3D(filters=filterNum, kernel_size=self.kSize, strides=strSize,
                                  padding=padMode, data_format=self.data_format)
        self.BN = layers.BatchNormalization(axis=self.conAx)
        self.Relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.BN(x, training=training)
        outputs = self.Relu(x)
        return outputs

    def build_model(self, input_shape):
        '''
        Helper method to build the model by running a dummy forward pass.
        input_shape: tuple, shape including batch dim e.g. (batch_size, depth, height, width, channels)
        '''
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs, training=True)


class lossExample(tf.keras.losses.Loss):
    '''
    A minimal custom loss subclass example.
    This subclass does not have any variables, so
    it does not introduce additional checkpoint-tracked state.
    '''
    def __init__(self, name='lossExample'):
        super().__init__(name=name)

    def call(self, y_pred, y_true):
        # Example loss simply mean difference (not meaningful loss, just placeholder)
        return tf.reduce_mean(y_pred - y_true)


class SoftDiceLoss(tf.keras.losses.Loss):
    '''
    SoftDiceLoss calculates multi-class soft dice loss for 5-D tensors.
    Inputs yTrue and yPred shaped like [mbSize, classNum, dim1, dim2, dim3].
    This class currently has no tracked variables, so it cannot be checkpointed.
    '''
    def __init__(self, wPow=2.0, name='SoftDiceLoss'):
        super().__init__(name=name)
        self.epsilon = 1e-16
        self.wPow = wPow

    def call(self, y_pred, y_true):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # Elementwise multiplication and sum over spatial dims (assumed dims 2,3,4)
        crossProd = y_pred * y_true
        crossProdSum = tf.reduce_sum(crossProd, axis=[2, 3, 4])
        weight = tf.reduce_sum(y_true, axis=[2, 3, 4])
        weight = 1 / (tf.pow(weight, self.wPow) + self.epsilon)
        numerator = 2 * tf.reduce_sum(crossProdSum * weight, axis=1)
        yySum = tf.reduce_sum(y_pred**2 + y_true**2, axis=[2, 3, 4])
        denominator = tf.reduce_sum(weight * yySum, axis=1)
        loss = tf.reduce_mean(1 - numerator / (denominator + self.epsilon))
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({'wPow': self.wPow})
        return config


class MyModel(tf.keras.Model):
    '''
    Fused model that contains:
    - A 3D Conv + BN + ReLU submodel (C3BR)
    - And exposes two loss functions: SoftDiceLoss and lossExample
    
    Demonstrates how a composite model might encapsulate
    the original model and custom loss objects.

    Forward pass returns a dictionary with:
    - "features": output tensor of C3BR
    - "dice_loss": soft dice loss computed between features and a dummy target (for demo)
    - "example_loss": example loss computed similarly

    For real use, losses should be computed during training with true labels.
    Here, we illustrate structural fusion per the issue context.
    '''

    def __init__(self):
        super().__init__()
        # Instantiate the 3D convolutional block with sample parameters
        # Using 'channels_first' to match axis usage in loss as per original
        self.submodel = C3BR(filterNum=32, kSize=3, strSize=1, padMode='valid', dFormat='channels_first')
        self.soft_dice_loss = SoftDiceLoss(wPow=2.0)
        self.example_loss = lossExample()

    @tf.function  # ensure tf function and compatible with XLA jit_compile
    def call(self, inputs, labels=None, training=False):
        ''' Forward pass:
            inputs: tensor expected shape e.g. (batch, channels, depth, height, width)
            labels: dummy labels to compute losses if provided (same shape as outputs)
        '''
        features = self.submodel(inputs, training=training)
        outputs = {'features': features}
        if labels is not None:
            # Compute losses if labels provided
            dice_loss_value = self.soft_dice_loss(features, labels)
            example_loss_value = self.example_loss(features, labels)
            outputs['dice_loss'] = dice_loss_value
            outputs['example_loss'] = example_loss_value
        return outputs


def my_model_function():
    # Return an instance of MyModel with all submodules initialized
    model = MyModel()
    # For proper shape inference and build, provide a dummy input shape
    # Assuming input shape: (batch=1, channels=1, depth=64, height=64, width=64)
    # to conform with channels_first format used in C3BR
    model.submodel.build_model(input_shape=(1, 1, 64, 64, 64))
    return model


def GetInput():
    # Return random input tensor that matches MyModel input in channels_first format
    # shape: (batch_size=1, channels=1, 64, 64, 64)
    return tf.random.uniform((1, 1, 64, 64, 64), dtype=tf.float32)

