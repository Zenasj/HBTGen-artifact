# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

def myconv2d(ix, w, padding):
    # Implements a custom Conv2D using tf.image.extract_patches
    # ix: input tensor of shape [B, H, W, C_in]
    # w: weight tensor of shape [filter_height, filter_width, in_channels, out_channels]
    filter_height = int(w.shape[0])
    filter_width = int(w.shape[1])
    in_channels = int(w.shape[2])
    out_channels = int(w.shape[3])
    ix_height = int(ix.shape[1])
    ix_width = int(ix.shape[2])
    ix_channels = int(ix.shape[3])
    # Extract patches
    patches = tf.image.extract_patches(
        images=ix,
        sizes=[1, filter_height, filter_width, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding=padding
    )
    # patches shape: [B, H_out, W_out, filter_height * filter_width * in_channels]
    # Reshape for elementwise multiply with flattened weights for each filter
    patches_reshaped = patches  # Shape already [B, H_out, W_out, filter_height * filter_width * in_channels]
    flat_w = tf.reshape(w, [filter_height * filter_width * in_channels, out_channels])  # [K, out_channels]

    # For each filter (out_channel), multiply patches with weights and sum
    feature_maps = []
    for i in range(out_channels):
        # Multiply patches with weights for i-th filter across the channel dim
        # broadcast flat_w[:, i] [K,] over patches_reshaped last dim [K]
        weighted = patches_reshaped * flat_w[:, i]
        feature_map = tf.reduce_sum(weighted, axis=3, keepdims=True)  # sum over patch dimension
        feature_maps.append(feature_map)
    features = tf.concat(feature_maps, axis=3) # [B, H_out, W_out, out_channels]
    return features


class MyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='SAME', **kwargs):
        super(MyConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
        })
        return config

    def build(self, input_shape):
        # kernel shape: (filter_height, filter_width, input_channels, filters)
        shape = self.kernel_size + (int(input_shape[-1]), self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='bias', shape=(self.filters,),
                                 initializer='random_normal', trainable=True)
        super(MyConv2D, self).build(input_shape)

    def call(self, inputs):
        conv_out = myconv2d(inputs, self.kernel, self.padding)
        return conv_out + self.b

    def compute_output_shape(self, input_shape):
        # Output shape same height and width due to padding
        return input_shape[:-1] + (self.filters,)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using the custom Conv2D layer with 32 filters, 3x3 kernel and 'SAME' padding
        self.myconv = MyConv2D(filters=32, kernel_size=(3, 3), padding='SAME')
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.myconv(inputs)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    model = MyModel()
    # Build model by calling on sample input to create weights
    dummy_input = GetInput()
    model(dummy_input)
    # Compile model similarly to original
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def GetInput():
    # Generate random input tensor matching MNIST images: batch size 1, shape 28x28, 1 channel, float32
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

