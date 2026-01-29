# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)

import tensorflow as tf
from distutils.version import LooseVersion

class MyModel(tf.keras.Model):
    """
    This implements the ResNet CIFAR-10 model with a custom CW loss.
    The model expects two inputs: images (batch, 32, 32, 3) and one-hot labels (batch, 10).
    """
    def __init__(self, img_h=32, img_w=32, channels=3, num_classes=10, n=8):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels
        self.num_classes = num_classes
        self.n = n  # Number of residual blocks per stack
        self.num_filters = 16  # Initial number of filters

        # Build layers for the ResNet model (following https://keras.io/examples/cifar10_resnet/)
        self._build_model()

    def _resnet_layer(self, num_filters=16, kernel_size=3, strides=1,
                      activation='relu', batch_normalization=True, conv_first=True):
        """
        Returns a keras sequential model layer that applies Conv-BatchNorm-Activation in order or
        BatchNorm-Activation-Conv order depending on conv_first.
        """
        layers = []
        layers.append(
            tf.keras.layers.Conv2D(num_filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        )
        if batch_normalization:
            layers.append(tf.keras.layers.BatchNormalization())
        if activation is not None:
            layers.append(tf.keras.layers.Activation(activation))

        def layer_func(x):
            if conv_first:
                for layer in layers:
                    x = layer(x)
                return x
            else:
                # Reverse order: BatchNorm-Activation-Conv
                if batch_normalization:
                    x = layers[1](x)
                if activation is not None:
                    x = layers[2](x) if batch_normalization else layers[1](x)
                x = layers[0](x)
                return x

        return layer_func

    def _build_model(self):
        """
        Builds the internal ResNet model as a keras Functional API model.
        The model takes two inputs: images and label_ref (one-hot labels) to be used in cw_loss.
        """
        # Input placeholders (symbolic tensors)
        self.clf_input = tf.keras.layers.Input(shape=(self.img_h, self.img_w, self.channels), name="model/input")
        self.label_ref = tf.keras.layers.Input(shape=(self.num_classes,), name='label_ref')

        x = self.clf_input
        num_filters = self.num_filters
        n = self.n

        # First conv layer stack
        x = self._resnet_layer(num_filters=num_filters)(x)
        for stack in range(3):
            for res_block in range(n):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2  # downsample at start of stack > 0
                y = self._resnet_layer(num_filters=num_filters, strides=strides)(x)
                y = self._resnet_layer(num_filters=num_filters, activation=None)(y)
                if stack > 0 and res_block == 0:
                    x = self._resnet_layer(num_filters=num_filters,
                                           kernel_size=1,
                                           strides=strides,
                                           activation=None,
                                           batch_normalization=False)(x)
                x = tf.keras.layers.Add()([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2
        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        x = tf.keras.layers.Flatten()(x)
        self.logits = tf.keras.layers.Dense(self.num_classes, kernel_initializer='he_normal', activation=None)(x)
        self.probs = tf.keras.layers.Activation('softmax')(self.logits)

        # For use as Functional model internally
        self.model = tf.keras.Model(inputs=[self.clf_input, self.label_ref], outputs=self.probs, name='clf_model')

        # Custom cw_loss function uses logits and label_ref
        def cw_loss(y_true, y_pred):
            """
            Carlini-Wagner like loss:
            y_true is label_ref input placeholder (one-hot labels),
            y_pred is softmax output (not used in cw_loss),
            logits are accessed from self.logits functional tensor.
            """
            label_mask = self.label_ref  # symbolic input
            pre_softmax = self.logits  # raw logits from last dense layer

            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keep_dims=True)
            else:
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keepdims=True)

            # Compute distance between logits of classes and correct class logits + margin
            distance = tf.nn.relu(pre_softmax - correct_logit + (1 - label_mask) * 10)

            # Inactivate threshold mask
            inactivate = tf.cast(tf.less_equal(distance, 1e-9), dtype=tf.float32)
            # Large negative weight to inactivate entries below threshold, then softmax as weighting
            weight = tf.keras.layers.Activation('softmax')(-1e9 * inactivate + distance)

            loss = tf.reduce_sum((1 - label_mask) * distance * weight, axis=1)
            loss = tf.math.reduce_mean(loss)
            return loss

        # Compile the internal keras model with Adam optimizer,
        # Use categorical_crossentropy loss by default,
        # Provide cw_loss as a metric for monitoring
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', cw_loss]
        )

    def call(self, inputs, training=False):
        """
        Forward pass with inputs = tuple (images, labels)
        - images: float32 tensor with shape (batch, 32, 32, 3)
        - labels: one-hot tensor with shape (batch, 10)
        Returns predicted softmax probabilities.
        """
        images, labels = inputs
        return self.model([images, labels], training=training)

def my_model_function():
    """
    Instantiate MyModel with default CIFAR10 params, return the model.
    """
    return MyModel()

def GetInput():
    """
    Generate a random batch input tuple (images, labels) compatible with MyModel.
    Images: random floats in [0, 1], shape (batch_size, 32, 32, 3)
    Labels: random one-hot vectors of length 10, shape (batch_size, 10)
    """
    batch_size = 100
    img_h = 32
    img_w = 32
    channels = 3
    num_classes = 10

    images = tf.random.uniform((batch_size, img_h, img_w, channels), minval=0., maxval=1., dtype=tf.float32)
    # Random integer labels from 0 to 9
    labels_int = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_int, depth=num_classes, dtype=tf.float32)

    return (images, labels_onehot)

