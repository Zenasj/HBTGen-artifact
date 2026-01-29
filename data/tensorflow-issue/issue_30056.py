# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, Input, Layer, PReLU
from tensorflow.keras.regularizers import l2

# Assumptions / Notes:
# - Input shape is MNIST images: (batch_size, 28, 28, 1)
# - The model includes a custom center loss layer that uses variable scatter_sub operation
# - scatter_sub is replaced with a workaround compatible with multi-GPU MirroredStrategy:
#   Using tf.Variable.scatter_sub directly instead of tf.compat.v1.scatter_sub.
# - The class MyModel fuses the base CNN and the center loss logic.
# - The model outputs two tensors: main classification output (10 classes) and center loss output (shape (batch_size, 1))


class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, num_classes=10, feat_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def build(self, input_shape):
        # centers is a non-trainable variable updated manually
        self.centers = self.add_weight(
            name='centers',
            shape=(self.num_classes, self.feat_dim),
            initializer='uniform',
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        features, labels = inputs
        labels = tf.reshape(labels, [-1])
        features = tf.cast(features, tf.float32)

        # Gather centers corresponding to each label
        centers_batch = tf.gather(self.centers, labels)

        # Count occurrences per label in batch
        unique_labels, unique_idx, unique_counts = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_counts, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        delta_centers = centers_batch - features
        delta_centers = delta_centers / tf.cast((1 + appear_times), tf.float32)
        delta_centers = self.alpha * delta_centers

        # Workaround for multi-GPU: use assign_sub on variable directly instead of tf.compat.v1.scatter_sub
        # But scatter_sub requires indices and updates; self.centers is variable of shape (num_classes, feat_dim)
        # indices: labels, updates: delta_centers
        # Equivalent is to use tf.tensor_scatter_nd_sub or the variable's scatter_sub method for each class.
        # We'll do a manual update loop on unique labels to avoid scatter_sub issues.

        def update_center(label, delta):
            # update one center vector at index label by subtracting delta
            delta_expanded = tf.expand_dims(delta, axis=0)  # shape (1, feat_dim)
            self.centers.scatter_sub(tf.IndexedSlices(delta_expanded[0], [label]))
            return 0

        # Aggregate deltas per label by summation because multiple samples might share label
        sum_deltas = tf.math.unsorted_segment_sum(delta_centers, labels, self.num_classes)

        # Update centers for each unique label
        def body(i, _):
            label = unique_labels[i]
            delta = sum_deltas[label]
            self.centers[label].assign_sub(delta)
            return i + 1, 0

        # Use tf.range + tf.function loop instead of python loop for graph compatibility
        i = tf.constant(0)

        # tf.while_loop to iterate over unique labels
        def cond(i, _):
            return i < tf.size(unique_labels)

        def body_loop(i, _):
            label = unique_labels[i]
            delta = sum_deltas[label]
            self.centers[label].assign_sub(delta)
            return i + 1, 0

        tf.while_loop(cond, body_loop, [i, 0])

        # Calculate the output which is the center loss term per sample (||features - centers_batch||^2)
        self.result = tf.reduce_sum(tf.square(features - centers_batch), axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, 1)
        return (input_shape[0][0], 1)


class MyModel(tf.keras.Model):
    def __init__(self, weight_decay=0.0005, num_classes=10, center_loss_alpha=0.5):
        super().__init__()
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.center_loss_alpha = center_loss_alpha

        # CNN layers as in my_model function from the issue
        self.conv1a = Conv2D(32, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn1a = BatchNormalization()
        self.prelu1a = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.conv1b = Conv2D(32, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn1b = BatchNormalization()
        self.prelu1b = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.pool1 = MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.conv2a = Conv2D(64, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn2a = BatchNormalization()
        self.prelu2a = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.conv2b = Conv2D(64, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn2b = BatchNormalization()
        self.prelu2b = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.pool2 = MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.conv3a = Conv2D(128, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn3a = BatchNormalization()
        self.prelu3a = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.conv3b = Conv2D(128, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(self.weight_decay))
        self.bn3b = BatchNormalization()
        self.prelu3b = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

        self.pool3 = MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()

        # Dense output features for center loss (dim 2), shared side output
        self.dense_feat = Dense(2, kernel_regularizer=l2(self.weight_decay))
        self.side_prelu = PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25), name='side_out')

        # Classification output
        self.classifier = Dense(self.num_classes, activation='softmax', name='main_out', kernel_regularizer=l2(self.weight_decay))

        # Center loss layer
        self.center_loss_layer = CenterLossLayer(alpha=self.center_loss_alpha, num_classes=self.num_classes, feat_dim=2, name='centerlosslayer')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs: tuple (x, labels)
        x, labels = inputs
        # Full forward pass

        # Block 1
        x = self.conv1a(x)
        x = self.bn1a(x, training=training)
        x = self.prelu1a(x)

        x = self.conv1b(x)
        x = self.bn1b(x, training=training)
        x = self.prelu1b(x)

        x = self.pool1(x)

        # Block 2
        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = self.prelu2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.prelu2b(x)

        x = self.pool2(x)

        # Block 3
        x = self.conv3a(x)
        x = self.bn3a(x, training=training)
        x = self.prelu3a(x)

        x = self.conv3b(x)
        x = self.bn3b(x, training=training)
        x = self.prelu3b(x)

        x = self.pool3(x)

        x = self.dropout(x, training=training)
        x = self.flatten(x)

        # Feature output for center loss
        feat = self.dense_feat(x)
        feat = self.side_prelu(feat)

        # Classification output
        main_out = self.classifier(feat)

        # Center loss output
        side_out = self.center_loss_layer((feat, labels))

        return main_out, side_out


def my_model_function():
    # Returns a MyModel instance
    return MyModel()


def GetInput():
    # Return a tuple of inputs matching MyModel expected input: (image batch, labels batch)
    # We use batch size 64 as in the original script and MNIST 28x28x1 input shape.
    batch_size = 64
    img_shape = (28, 28, 1)
    # Features: random float tensor simulating MNIST normalized images 0..1
    images = tf.random.uniform((batch_size,) + img_shape, minval=0, maxval=1, dtype=tf.float32)
    # Labels: random ints in [0,9]
    labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)
    return (images, labels)

