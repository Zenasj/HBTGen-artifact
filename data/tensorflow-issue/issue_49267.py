# tf.random.uniform((B, H, W, C), dtype=tf.float32)
import tensorflow as tf

class Localization(tf.keras.layers.Layer):
    def __init__(self):
        super(Localization, self).__init__()
        self.bpool1 = tf.keras.layers.MaxPool2D()
        self.bpool2 = tf.keras.layers.MaxPool2D()
        self.bpool3 = tf.keras.layers.MaxPool2D()
        self.bpool4 = tf.keras.layers.MaxPool2D()

        self.mpool1 = tf.keras.layers.MaxPool2D()
        self.mpool2 = tf.keras.layers.MaxPool2D()
        self.mpool3 = tf.keras.layers.MaxPool2D()
        self.mpool4 = tf.keras.layers.MaxPool2D()

        self.cpool1 = tf.keras.layers.MaxPool2D()
        self.cpool2 = tf.keras.layers.MaxPool2D()
        self.cpool3 = tf.keras.layers.MaxPool2D()
        self.cpool4 = tf.keras.layers.MaxPool2D()

        self.bconv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.bconv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.bconv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.bconv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.bconv5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.bconv6 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.bconv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.bconv8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')

        self.mconv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.mconv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.mconv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.mconv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.mconv5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.mconv6 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.mconv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.mconv8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')

        self.cconv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.cconv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.cconv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.cconv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.cconv5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.cconv6 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.cconv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.cconv8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')

        # Fix: Use functional concatenate, not assignment of the function directly
        self.concatenate1 = tf.keras.layers.Concatenate()
        self.concatenate2 = tf.keras.layers.Concatenate()
        self.tconv1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.tpool1 = tf.keras.layers.MaxPool2D()
        self.tconv2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.tpool2 = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc0 = tf.keras.layers.Dense(100, activation='relu')
        self.fc1 = tf.keras.layers.Dense(20, activation='relu')
        self.fc2 = tf.keras.layers.Dense(6, 
                 activation=None, 
                 bias_initializer=tf.keras.initializers.Constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 
                 kernel_initializer='zeros',  
                 activity_regularizer=tf.keras.regularizers.l2(1e-2))

    def call(self, inputs):
        mask, fg, bg, composite = inputs
        xm = self.concatenate1([fg, mask])  # corrected usage
        xm = self.mconv1(xm)
        xm = self.mconv2(xm)
        xm = self.mpool1(xm)

        xm = self.mconv3(xm)
        xm = self.mconv4(xm)
        xm = self.mpool2(xm)

        xm = self.mconv5(xm)
        xm = self.mconv6(xm)
        xm = self.mpool3(xm)

        xm = self.mconv7(xm)
        xm = self.mconv8(xm)
        xm = self.mpool4(xm)

        xbg = self.bconv1(bg)
        xbg = self.bconv2(xbg)
        xbg = self.bpool1(xbg)

        xbg = self.bconv3(xbg)
        xbg = self.bconv4(xbg)
        xbg = self.bpool2(xbg)

        xbg = self.bconv5(xbg)
        xbg = self.bconv6(xbg)
        xbg = self.bpool3(xbg)

        xbg = self.bconv7(xbg)
        xbg = self.bconv8(xbg)
        xbg = self.bpool4(xbg)

        xc = self.cconv1(composite)
        xc = self.cconv2(xc)
        xc = self.cpool1(xc)

        xc = self.cconv3(xc)
        xc = self.cconv4(xc)
        xc = self.cpool2(xc)

        xc = self.cconv5(xc)
        xc = self.cconv6(xc)
        xc = self.cpool3(xc)

        xc = self.cconv7(xc)
        xc = self.cconv8(xc)
        xc = self.cpool4(xc)

        x = self.concatenate2([xbg, xm, xc])  # corrected usage
        x = self.tconv1(x)
        x = self.tpool1(x)
        x = self.tconv2(x)
        x = self.tpool2(x)  # corrected: use tpool2 once (not tpool1 twice)
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        theta = self.fc2(x)
        theta = tf.reshape(theta, (-1, 2, 3))  # reshape layer inside call to avoid overhead
        return theta

class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=320, width=320):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, input_shape[0][-1]]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }

    def advance_indexing(self, inputs, x, y):
        '''
        TensorFlow workaround for numpy-like advanced indexing.
        '''
        shape = tf.shape(inputs)
        batch_size = shape[0]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], axis=3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        batch = tf.shape(images)[0]
        homogenous_coordinates = self.grid_generator(batch)
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1.0, 1.0, self.width)
        y = tf.linspace(-1.0, 1.0, self.height)
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates

    def interpolate(self, images, homogenous_coordinates, theta):
        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])

            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]

            x = ((x_transformed + 1.0) * tf.cast(self.width, tf.float32)) * 0.5
            y = ((y_transformed + 1.0) * tf.cast(self.height, tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width - 1)
            x1 = tf.clip_by_value(x1, 0, self.width - 1)
            y0 = tf.clip_by_value(y0, 0, self.height - 1)
            y1 = tf.clip_by_value(y1, 0, self.height - 1)

            x = tf.clip_by_value(x, 0.0, tf.cast(self.width, tf.float32) - 1.0)
            y = tf.clip_by_value(y, 0.0, tf.cast(self.height, tf.float32) - 1.0)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, tf.float32)
            x1 = tf.cast(x1, tf.float32)
            y0 = tf.cast(y0, tf.float32)
            y1 = tf.cast(y1, tf.float32)

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

class Composition(tf.keras.layers.Layer):
    def __init__(self):
        super(Composition, self).__init__()

    def call(self, inputs):
        mask, fg, bg = inputs
        multiples = tf.constant([1, 1, 1, 3], tf.int32)
        mask_mod = tf.tile(mask, multiples)
        bg_mod = tf.keras.layers.Multiply()([bg, 1 - mask_mod])
        fg_mod = tf.keras.layers.Multiply()([fg, mask_mod])
        composite_image = tf.keras.layers.Add()([bg_mod, fg_mod])
        return composite_image

class MyModel(tf.keras.Model):
    def __init__(self, bg_shape=(420, 640, 3), iterations=5):
        super(MyModel, self).__init__()
        self.bg_shape = bg_shape
        self.iterations = iterations
        self.localize = Localization()
        self.bilinearintp1 = BilinearInterpolation(height=self.bg_shape[0], width=self.bg_shape[1])
        self.bilinearintp2 = BilinearInterpolation(height=self.bg_shape[0], width=self.bg_shape[1])
        self.compose1 = Composition()
        self.compose2 = Composition()

    def call(self, inputs):
        # inputs: mask, fg, bg tensors
        mask, fg, bg = inputs
        xmask = mask
        xfg = fg
        composite = self.compose1([xmask, xfg, bg])
        for _ in range(self.iterations):
            theta = self.localize([xmask, xfg, bg, composite])
            xmask = self.bilinearintp1([xmask, theta])
            xfg = self.bilinearintp2([xfg, theta])
            composite = self.compose2([xmask, xfg, bg])
        return composite

def my_model_function():
    # Instantiate MyModel with default parameters
    return MyModel()

def GetInput():
    # Returns a tuple of (mask, fg, bg) inputs compatible with MyModel
    # Assuming height=420, width=640, 3 channels for fg and bg, mask is single channel (height,width,1)
    batch_size = 1
    height = 420
    width = 640
    channels = 3

    # Random mask: shape (B, H, W, 1), values [0,1]
    mask = tf.random.uniform((batch_size, height, width, 1), minval=0, maxval=1, dtype=tf.float32)
    # Random fg and bg images: shape (B, H, W, C), values [0,1]
    fg = tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=1, dtype=tf.float32)
    bg = tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=1, dtype=tf.float32)
    return (mask, fg, bg)

