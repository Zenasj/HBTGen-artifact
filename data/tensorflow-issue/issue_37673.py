# tf.random.uniform((BATCH_SIZE, 2), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Input

class MyModel(tf.keras.Model):
    def __init__(self, batchnorm_momentum=0.9):
        super().__init__()
        self.batchnorm_momentum = batchnorm_momentum
        # Generator layers - 4 layers of Dense(64) + relu, final Dense(2)
        self.g_layers = []
        for i in range(4):
            # Note: reuse input shape (2,)
            self.g_layers.append(Dense(64))
            self.g_layers.append(Activation("relu"))
        self.g_layers.append(Dense(2))  # output layer
        
        # Discriminator layers - 4 layers of Dense(64) + optional BatchNorm + relu,
        # final Dense(1, sigmoid)
        self.d_layers = []
        for i in range(4):
            self.d_layers.append(Dense(64))
            # batchnorm possibly present
            if self.batchnorm_momentum:
                self.d_layers.append(BatchNormalization(momentum=self.batchnorm_momentum))
            self.d_layers.append(Activation("relu"))
        self.d_layers.append(Dense(1, activation="sigmoid"))
        
        # The discriminator is not trainable in the GAN forward pass
        # but here for fused model we implement both forward separately

    def call(self, inputs, training=False):
        """Forward pass:
        inputs: noise tensor, shape (batch_size, 2).
        
        Returns a dict with keys:
          'generator_output': tensor (batch_size,2),
          'discriminator_on_real': None (not provided here),
          'discriminator_on_fake': sigmoid output on generated samples,
          'gan_output': discriminator output on generator output,
          'comparison': boolean tensor comparing the D outputs in some way (optional).
          
        Since the original issue shows separate models, here we implement
        generator -> discriminator path and return outputs.
        
        Note: In usual GAN training, discriminator is trained separately on
        real samples and fake samples; generator inputs noise and is trained
        via discriminator outputs.
        
        This fused model processes noise input through G, then through D,
        returning discriminator output on generated fake samples.
        """
        x = inputs
        # Generator forward
        for layer in self.g_layers:
            # When layer is BatchNorm, pass training flag
            if isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        gen_output = x  # Generator output (fake samples)
        
        # Discriminator forward on generated samples
        y = gen_output
        for layer in self.d_layers:
            if isinstance(layer, BatchNormalization):
                y = layer(y, training=training)
            else:
                y = layer(y)
        d_fake_output = y  # Discriminator prediction on generator output
        
        # As we cannot input real samples here (fused single-input model),
        # we cannot get discriminator real outputs; for demonstration
        # we only return discriminator on generated data
        
        return {
            'generator_output': gen_output,
            'discriminator_on_fake': d_fake_output,
        }

def my_model_function():
    # Return an instance of MyModel; default batchnorm enabled (momentum=0.9)
    # This matches the reported problematic config with batchnorm.
    return MyModel(batchnorm_momentum=0.9)

def GetInput():
    # Return random noise input tensor: shape (batch_size, 2)
    batch_size = 32  # typical batch size used in example training
    # Values uniform between -1 and 1 as in get_noise(num)
    return tf.random.uniform((batch_size, 2), minval=-1.0, maxval=1.0, dtype=tf.float32)

