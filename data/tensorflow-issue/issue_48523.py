# tf.random.uniform((1, 300, 300, 3), dtype=tf.float32) ← inferred input shape and dtype from create_demo()

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base MobileNetV2 feature extractor (up to layer "out_relu")
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(300, 300, 3), pooling=None
        )
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(name="out_relu").output,
            name="feature_extractor"
        )
        
        # Conv-BN-Softmax block after feature extractor
        self.demo_conv01 = tf.keras.layers.Conv2D(
            4, (3, 3), padding="same", strides=(1, 1), kernel_initializer="he_uniform", name="demo_conv01"
        )
        self.demo_bn01 = tf.keras.layers.BatchNormalization(name="demo_bn01")
        self.demo_softmax01 = tf.keras.layers.Softmax(axis=2, name="a_demo_softmax01")
        
        # Layers to slice out each channel from softmax
        self.demo_slice_layers = [
            tf.keras.layers.Lambda(lambda x, i=i: x[:, :, :, i], name=f"demo_slice0{i+1}") for i in range(4)
        ]
        # Reshape layers for slices to [10, 10, 1]
        self.demo_slice_reshapes = [
            tf.keras.layers.Reshape([10, 10, 1], name=f"demo_slice0{i+1}_reshape") for i in range(4)
        ]
        
        # Block layers reused in four branches:
        self.demo_block_conv01 = tf.keras.layers.Conv2D(
            3, (3, 3), padding="same", strides=(1, 1), kernel_initializer="he_uniform", name="demo_block_conv01"
        )
        self.demo_block_bn01 = tf.keras.layers.BatchNormalization(name="demo_block_bn01")
        self.demo_block_relu01 = tf.keras.layers.ReLU(name="demo_block_relu01")
        self.demo_block_flatten = tf.keras.layers.Flatten(name="demo_block_flatten")
        self.demo_block_softmax01 = tf.keras.layers.Dense(4, activation="softmax", name="demo_block_softmax01")
        
        # Multiply layers for four branches
        self.demo_mul_names = [
            "w_demo_mul01",
            "x_demo_mul02",
            "y_demo_mul03",
            "z_demo_mul04"
        ]
        self.demo_mul_layers = [
            tf.keras.layers.Multiply(name=name) for name in self.demo_mul_names
        ]
        
        # Layers to wrap block outputs - these just pass through block outputs
        # As tf.keras.layers.Layer() is a base abstract, replace by Lambda layers for identity
        self.demo_class_layers = [
            tf.keras.layers.Lambda(lambda x: x, name=f"demo_class0{i+1}") for i in range(4)
        ]
    
    def call(self, inputs, training=False):
        # Extract features from MobileNetV2 backbone
        out_relu = self.feature_extractor(inputs, training=training)  # shape ~ (1, 10, 10, 1280)
        
        # Conv-BN-Softmax block
        x = self.demo_conv01(out_relu)
        x = self.demo_bn01(x, training=training)
        demo_softmax01 = self.demo_softmax01(x)
        
        # Slice and reshape each channel from softmax output
        demo_slices = []
        demo_slices_reshaped = []
        for slice_layer, reshape_layer in zip(self.demo_slice_layers, self.demo_slice_reshapes):
            slice_i = slice_layer(demo_softmax01)  # shape (1, 10, 10)
            demo_slices.append(slice_i)
            slice_reshaped_i = reshape_layer(slice_i)  # shape (1,10,10,1)
            demo_slices_reshaped.append(slice_reshaped_i)
        
        # Multiply out_relu by each sliced channel mask
        demo_mul_outputs = []
        for mul_layer, slice_r in zip(self.demo_mul_layers, demo_slices_reshaped):
            mul_out = mul_layer([out_relu, slice_r])  # broadcast multiply (1,10,10,1280) * (1,10,10,1)
            demo_mul_outputs.append(mul_out)
        
        # Block function: Conv-BN-ReLU-Flatten-Dense(softmax)
        def block_branch(x):
            c = self.demo_block_conv01(x)
            b = self.demo_block_bn01(c, training=training)
            r = self.demo_block_relu01(b)
            f = self.demo_block_flatten(r)
            s = self.demo_block_softmax01(f)
            return s  # shape (1,4)
        
        # Get class predictions per branch using block
        demo_class_outputs = [layer(block_branch(mul_out)) for layer, mul_out in zip(self.demo_class_layers, demo_mul_outputs)]
        
        # Return outputs consistent with original model outputs:
        # [demo_softmax01, demo_class01, demo_class02, demo_class03, demo_class04, demo_mul01, demo_mul02, demo_mul03, demo_mul04]
        return [demo_softmax01] + demo_class_outputs + demo_mul_outputs


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    
    # Build the model by calling once with dummy input to initialize weights
    dummy_input = tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    return model


def GetInput():
    # Return a random tensor input that matches expected input shape of MyModel
    # Tensor shape: (batch=1, height=300, width=300, channels=3), dtype float32 typical for MobileNetV2
    return tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)

