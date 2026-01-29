# tf.random.uniform((1, 8, 4), dtype=tf.float32) ← This shape is based on example input used with TimeDistributed wrapping a layer with input shape (batch, timesteps, features)

import tensorflow as tf

class TestLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # The mask argument is optional here to accommodate TimeDistributed's rnn-version calling.
        # Sum along the last dimension and multiply by mask if provided; else propagate input sum only.
        summed = tf.reduce_sum(inputs, axis=-1)
        if mask is not None:
            mask = tf.cast(mask, summed.dtype)
            # The mask shape may vary: if 2D ([batch, timestep]) it must be aligned carefully.
            # TimeDistributed treats mask as [batch, timestep], inputs as [batch, timestep, features].
            # Since TestLayer is called with a timestep slice in rnn-mode (input shape [batch, features]), mask is generally None.
            # But if mask present, multiply accordingly.

            # To robustly handle both modes, try elementwise multiply broadcasted:
            # summed shape: [batch, timestep] or [batch], depending on mode
            # mask shape may be [batch, timestep] or [batch]
            # So broadcasting multiply should suffice.
            return summed * mask
        else:
            return summed

class ClonedGlobalAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self):
        super(ClonedGlobalAveragePooling1D, self).__init__()        
        self.supports_masking = True
        
    def build(self, input_shape):
        self.internal_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.internal_layer.build(input_shape)

    def call(self, inputs, mask=None):
        return self.internal_layer(inputs, mask=mask)
    
    def compute_mask(self, inputs, mask=None):
        return self.internal_layer.compute_mask(inputs, mask=mask)
        
    def compute_output_shape(self, input_shape):
        return self.internal_layer.compute_output_shape(input_shape)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.test_layer = tf.keras.layers.TimeDistributed(TestLayer())
        self.cloned_gap = tf.keras.layers.TimeDistributed(ClonedGlobalAveragePooling1D())
        self.gap_direct = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())
        
        # By default, _always_use_reshape is False leading to rnn-version usage in TimeDistributed.
        # Set _always_use_reshape=True for the test_layer and gap layers to enable non-rnn reshape behavior 
        # that passes masks as expected to the wrapped custom layers.
        # Note that in practice, _always_use_reshape is a private property, but used here to illustrate the difference.
        self.test_layer._always_use_reshape = True
        self.cloned_gap._always_use_reshape = True
        self.gap_direct._always_use_reshape = True

    def call(self, inputs, mask=None):
        """
        inputs: expected shape [batch, timesteps, features, channels] for GAP layers,
                and [batch, timesteps, features] for TestLayer.
        mask: expected shape [batch, timesteps]. This mask is propagated by TimeDistributed.
        """
        # Forward through test_layer expects 3D input: [batch, timesteps, features]
        # Forward through GAP layers expects 4D input: [batch, timesteps, features, channels]

        # For demonstration, split inputs accordingly (infer from input shape)
        # Here assume inputs is a tuple (x_testlayer, x_gap)
        # If single input tensor, we can slice or assume padding dims.

        # For simplicity, assume input is a tuple of 2 tensors:
        # inputs[0]: for test_layer: shape [batch, timesteps, features]
        # inputs[1]: for gap layers: shape [batch, timesteps, features, channels]

        x_testlayer, x_gap = inputs

        # Call wrapped layers
        out_test = self.test_layer(x_testlayer, mask=mask)      # Shape: [batch, timesteps]
        out_cloned = self.cloned_gap(x_gap, mask=mask)          # Shape: [batch, timesteps, channels_removed]
        out_gap = self.gap_direct(x_gap, mask=mask)             # Shape: [batch, timesteps, channels_removed]

        # Demonstrate simple comparison between cloned and direct GAP layers' outputs with tolerance
        diff = tf.abs(out_cloned - out_gap)
        tolerance = 1e-6
        # Boolean tensor indicating if outputs are close elementwise
        close = tf.reduce_all(diff < tolerance)

        # Output a dict with:
        # - test_layer output
        # - direct GAP output
        # - cloned GAP output
        # - comparison result
        return {
            "test_layer_output": out_test,
            "gap_direct_output": out_gap,
            "cloned_gap_output": out_cloned,
            "outputs_match": close
        }

def my_model_function():
    return MyModel()

def GetInput():
    """
    Construct consistent inputs compatible with MyModel:
    - For test_layer: shape [batch=1, timesteps=8, features=4] as in original example
    - For gap layers: shape [batch=1, timesteps=8, features=4, channels=1] since GAP usually over last dim
    Return a tuple (input_for_testlayer, input_for_gap_layers)
    """
    batch = 1
    timesteps = 8
    features = 4
    channels = 1

    x_testlayer = tf.random.uniform((batch, timesteps, features), dtype=tf.float32)
    x_gap = tf.random.uniform((batch, timesteps, features, channels), dtype=tf.float32)

    return (x_testlayer, x_gap)

