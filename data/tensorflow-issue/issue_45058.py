# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming input shape typical for VGG19 (e.g. 224x224 RGB images)
import tensorflow as tf

def vgg_layers(layer_names):
    """
    Create a VGG19 model that outputs activations from specified layer names.
    Uses pretrained ImageNet weights. Model is non-trainable.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layers of interest from VGG19 for loss calculation:
        # Using conv1 layer of each block (block1_conv1 ... block5_conv1)
        self.layer_names = [f'block{i+1}_conv1' for i in range(5)]
        # Corresponding layer weights for weighted L1 loss:
        self.layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
        # VGG19 feature extractor:
        self.vgg = vgg_layers(self.layer_names)
        # L1 loss function:
        self.l1_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    
    def call(self, inputs):
        """
        Expected inputs: tuple (x, y) where both are image tensors.
        Computes weighted L1 loss between VGG19 feature activations of x and y.
        Returns a scalar tensor representing the VGGloss.
        """
        x, y = inputs
        # Preprocess inputs according to VGG19 preprocessing:
        # Convert from [0,1] or [0,255] RGB to BGR with mean subtraction.
        # Here, assuming inputs are in [0,1] float range:
        x_proc = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
        y_proc = tf.keras.applications.vgg19.preprocess_input(y * 255.0)

        x_features = self.vgg(x_proc)
        y_features = self.vgg(y_proc)

        loss = 0.0
        for w, xf, yf in zip(self.layer_weights, x_features, y_features):
            # Compute mean absolute error per layer and weight it
            # Note: we keep the sum reduction to get a scalar loss per layer
            layer_loss = self.l1_loss(xf, yf)
            loss += w * layer_loss
        return loss

def my_model_function():
    # Return an instance of MyModel, with VGG19 pretrained weights loaded internally.
    return MyModel()

def GetInput():
    """
    Returns a tuple of two float32 tensors representing image batches,
    shape (B, H, W, C) with batch size 2, height and width 224, channels 3.
    Values are uniform random floats in [0,1], suitable for VGG19 preprocessing.
    """
    batch_size = 2
    height = 224
    width = 224
    channels = 3
    x = tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)
    y = tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)
    return (x, y)

