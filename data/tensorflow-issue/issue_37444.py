# tf.random.uniform((1, 10, 10, 20), dtype=tf.float32) ← Input is a batch with shape [1, Height=10, Width=10, Channels=20]

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Fusion of the six Scaler* layers demonstrating different loop mechanisms to resize images
    inside tf.function and showing the differences that cause graph mode errors.

    The model takes a float tensor of shape [1, H, W, C], where H=10, W=10, C=20 by default,
    and applies multiple resizes with varying scales.

    The forward pass returns a dictionary mapping from scaler names to the list of resized images outputs
    from each approach.

    This fusion allows comparison between the various scaler implementations to understand behavior
    under tf.function/autograph and graph mode.
    """

    def __init__(self, count=5):
        super().__init__()
        # Store count as int for Python loops; avoid wrapping into tf.constant to prevent autograph issues
        self.count = count

        # Instantiate individual scalers
        self.scaler1 = Scaler1(count)
        self.scaler2 = Scaler2(count)
        self.scaler3 = Scaler3(count)
        self.scaler4 = Scaler4(count)
        self.scaler5 = Scaler5(count)
        self.scaler6 = Scaler6(count)

    @tf.function
    def call(self, inputs):
        # Run each Scaler and collect outputs
        out1 = self.scaler1(inputs)
        out2 = self.scaler2(inputs)
        out3 = self.scaler3(inputs)
        out4 = self.scaler4(inputs)
        out5 = self.scaler5(inputs)
        out6 = self.scaler6(inputs)

        # Return outputs as a dict mapping scaler names to list of resized images (tensors)
        return {
            'Scaler1': out1,
            'Scaler2': out2,
            'Scaler3': out3,
            'Scaler4': out4,
            'Scaler5': out5,
            'Scaler6': out6,
        }


class Scaler1(tf.keras.layers.Layer):
    """
    Original example using Python range converted count with append to Python list inside tf.function.
    This typically fails in graph mode due to Python list mutation inside tf.function with graph execution.
    """
    def __init__(self, count=5, name="Scaler1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count
        self.sized_images = []

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        self.sized_images = []

        for i in range(int(self.count)):
            scale = image_size * (1 + tf.cast(i, dtype=tf.float32))
            sized_image = tf.image.resize(images, tf.cast(scale + 0.5, tf.int32))
            self.sized_images.append(sized_image)

        return self.sized_images


class Scaler2(tf.keras.layers.Layer):
    """
    Uses tf.range for loop indices, but still uses Python list append inside tf.function.
    This can cause issues in graph mode (InaccessibleTensorError).
    """
    def __init__(self, count=5, name="Scaler2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count
        self.sized_images = []

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)
        self.sized_images = []

        for i in tf.range(self.count):
            scale = image_size * (1 + tf.cast(i, tf.float32))
            sized_image = tf.image.resize(images, tf.cast(scale + 0.5, tf.int32))
            self.sized_images.append(sized_image)

        return self.sized_images


class Scaler3(tf.keras.layers.Layer):
    """
    List comprehension inside tf.function with tf.range to build the list of resized images.
    This often works better because autograph can trace this pattern.
    """
    def __init__(self, count=5, name="Scaler3", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count
        self.sized_images = []

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)

        self.sized_images = [
            tf.image.resize(images, tf.cast(image_size * (1 + i) + 0.5, tf.int32))
            for i in tf.range(self.count)
        ]
        return self.sized_images


class Scaler4(tf.keras.layers.Layer):
    """
    Manually unrolled loop with nested if statements returning lists of resized images based on count.
    This approach is verbose but can avoid some autograph issues for small fixed counts.
    """
    def __init__(self, count=5, name="Scaler4", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)

        i = 1
        scale1 = image_size * (1 + tf.cast(i, tf.float32))
        sized_image1 = tf.image.resize(images, tf.cast(scale1 + 0.5, tf.int32))

        if i == self.count:
            return [sized_image1]
        else:
            i = 2
            scale2 = image_size * (1 + tf.cast(i, tf.float32))
            sized_image2 = tf.image.resize(images, tf.cast(scale2 + 0.5, tf.int32))

            if i == self.count:
                return [sized_image1, sized_image2]
            else:
                i = 3
                scale3 = image_size * (1 + tf.cast(i, tf.float32))
                sized_image3 = tf.image.resize(images, tf.cast(scale3 + 0.5, tf.int32))

                if i == self.count:
                    return [sized_image1, sized_image2, sized_image3]
                else:
                    i = 4
                    scale4 = image_size * (1 + tf.cast(i, tf.float32))
                    sized_image4 = tf.image.resize(images, tf.cast(scale4 + 0.5, tf.int32))

                    if i == self.count:
                        return [sized_image1, sized_image2, sized_image3, sized_image4]
                    else:
                        # For counts >4, return first 4 sized images as fallback
                        return [sized_image1, sized_image2, sized_image3, sized_image4]


class Scaler5(tf.keras.layers.Layer):
    """
    Uses tf.range loop and builds the list inside the loop but does NOT store to self.sized_images.
    This reduces state mutation and may behave slightly better in graph mode.
    """
    def __init__(self, count=5, name="Scaler5", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)

        sized_images = []
        for i in tf.range(self.count):
            scale = image_size * (1 + i)
            sized_images.append(tf.image.resize(images, tf.cast(scale + 0.5, tf.int32)))

        return sized_images


class Scaler6(tf.keras.layers.Layer):
    """
    Uses tf.TensorArray to collect scales then unpacks to list and applies resize.
    This approach is a recommended workaround to issues with list mutation in tf.function.
    """
    def __init__(self, count=5, name="Scaler6", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = count

    @tf.function
    def call(self, inputs):
        images = inputs
        image_size = tf.cast(tf.shape(images)[1:3], dtype=tf.float32)

        scales_arr = tf.TensorArray(dtype=tf.float32,
                                    size=tf.cast(self.count, dtype=tf.int32),
                                    dynamic_size=False)
        for i in tf.range(self.count):
            scale = image_size * (1 + i)
            scales_arr = scales_arr.write(tf.cast(i, tf.int32), scale)
        scales = scales_arr.stack()

        scales_list = tf.unstack(scales)
        sized_images = [tf.image.resize(images, tf.cast(scale + 0.5, tf.int32)) for scale in scales_list]

        return sized_images


def my_model_function():
    # Return instance of MyModel with default count=5
    return MyModel(count=5)


def GetInput():
    # Return a random float32 tensor with shape [1, 10, 10, 20] that fits MyModel input
    return tf.random.uniform((1, 10, 10, 20), dtype=tf.float32)

