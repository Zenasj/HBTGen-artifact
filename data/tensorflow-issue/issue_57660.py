# tf.random.normal((1, 1200, 1200, 3), dtype=tf.float32) and tf.constant((1, 4), dtype=tf.float32)

import tensorflow as tf

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_height=90, target_width=90, **kwargs):
        super().__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def resize_impl(self, elems):
        im, xy_x2y2 = elems
        x = xy_x2y2[0]
        y = xy_x2y2[1]
        width = xy_x2y2[2] - x
        height = xy_x2y2[3] - y
        # The original code comments an attempted crop - not doing here.
        # Resize with pad to fixed size
        im = tf.image.resize_with_pad(im, self.target_height, self.target_width)
        return im

    def call(self, x):
        im, coords = x
        coords = tf.cast(coords, tf.int32)
        # elems for map_fn: list/tuple of tensors (im, coords)
        # Shapes expected: im: (batch, H, W, 3), coords: (batch, 4)
        # tf.map_fn iterates over batches, so elems list has batch dimension
        # map_fn fn gets elems from both tensors at the same batch index
        return tf.map_fn(
            fn=self.resize_impl,
            elems=[im, coords],
            fn_output_signature=tf.TensorSpec((self.target_height, self.target_width, 3), dtype=tf.float32)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_height": self.target_height,
            "target_width": self.target_width,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MyModel(tf.keras.Model):
    def __init__(self, target_height=90, target_width=90, **kwargs):
        super().__init__(**kwargs)
        self.resize_layer = ResizeLayer(target_height=target_height, target_width=target_width)

    def call(self, inputs):
        # inputs is tuple (image_tensor, coords_tensor)
        # image_tensor shape: (batch, None, None, 3)
        # coords_tensor shape: (batch, 4)
        return self.resize_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "resize_layer": self.resize_layer.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        resize_conf = config.pop("resize_layer", {})
        model = cls(**config)
        model.resize_layer = ResizeLayer.from_config(resize_conf)
        return model


def my_model_function():
    # Create an instance of MyModel with default target size 90x90 as per original code
    return MyModel()

def GetInput():
    # Generate a random image input tensor and coords tensor compatible with MyModel
    # From original issue:
    # im shape: (1, 1200, 1200, 3) dtype float32
    # coords shape: (1, 4) float32 with some values around 100~1000
    im = tf.random.normal(shape=(1, 1200, 1200, 3), seed=42, dtype=tf.float32)

    # Using numpy random normal with loc and seed for coords as in issue
    import numpy as np
    np.random.seed(42)
    coords_np = np.array([[np.random.normal(loc=100),
                           np.random.normal(loc=100),
                           np.random.normal(loc=1000),
                           np.random.normal(loc=1000)]], dtype=np.float32)
    coords = tf.constant(coords_np)

    return (im, coords)

