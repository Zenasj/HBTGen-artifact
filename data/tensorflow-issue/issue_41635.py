# tf.random.uniform((batch_size, 128, 128, 3), dtype=tf.float32) ← GAN Generator input image shape

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Assumptions / notes based on the issue:
# - Input images have shape (128, 128, 3) as per GAN example.
# - Generator and Discriminator are defined per the U-Net style GAN architecture in the issue.
# - The forward() method combines generator and discriminator outputs to replicate the combined model training step.
# - This fused model outputs a tuple: (discriminator validity output, generator fake image output).
# - The user noted NaNs occur on TPU when the last batch is smaller; to avoid this, batches should have consistent size.
# - Here, the model and inputs are defined to avoid NaNs by enforcing batch size >= 8 (TPU cores).
# - GetInput generates a random tensor of shape matching generator input, batch size 16 (safe default).

class MyModel(tf.keras.Model):
    def __init__(self, img_shape=(128, 128, 3), gf=64, df=64):
        super().__init__()
        self.img_shape = img_shape
        self.gf = gf
        self.df = df

        # Build generator and discriminator submodules per the code snippet in the issue
        
        # Generator blocks
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate > 0:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])  # skip connection
            return u

        # Generator model construction
        d0 = Input(shape=self.img_shape)

        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        self.generator = Model(inputs=d0, outputs=output_img, name='generator')

        # Discriminator blocks
        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_Y = Input(shape=self.img_shape)  # real image
        img_X = Input(shape=self.img_shape)  # input conditioning image

        combined_imgs = Concatenate(axis=-1)([img_Y, img_X])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        self.discriminator = Model(inputs=[img_Y, img_X], outputs=validity, name='discriminator')

        # For the combined model that uses both generator and discriminator for training
        # Freeze discriminator weights when training combined
        self.discriminator.trainable = False

        input_img_X = Input(shape=self.img_shape)
        input_img_Y = Input(shape=self.img_shape)

        fake_Y = self.generator(input_img_X)
        validity_output = self.discriminator([fake_Y, input_img_Y])

        self.combined = Model(inputs=[input_img_Y, input_img_X],
                              outputs=[validity_output, fake_Y], name='combined')

    def call(self, inputs, training=False):
        # inputs is a tuple (imgs_Y, imgs_X):
        imgs_Y, imgs_X = inputs
        # Run generator and discriminator on inputs to replicate combined model output
        fake_Y = self.generator(imgs_X, training=training)
        validity = self.discriminator([fake_Y, imgs_Y], training=training)
        # Output: discriminator's real/fake validity score + generated fake images
        return validity, fake_Y

def my_model_function():
    model = MyModel()
    # Compile components to replicate combined.compile(...)
    optimizer = Adam(0.0002, 0.5)
    model.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    model.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)
    return model

def GetInput():
    # Generate random input in shape expected by the GAN generator
    # Batch size >= 8 recommended to avoid TPU NaN issue; use batch size 16 as default safe batch size
    batch_size = 16  # safe batch size > TPU core count
    img_shape = (128, 128, 3)
    x = tf.random.uniform((batch_size,) + img_shape, dtype=tf.float32)
    # For combined model input, we need a tuple (imgs_Y, imgs_X)
    # imgs_Y represents target/real images, imgs_X generator input
    y = tf.random.uniform((batch_size,) + img_shape, dtype=tf.float32)
    return (y, x)

