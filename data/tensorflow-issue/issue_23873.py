from typing import Tuple, Callable, Any, Optional
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.training.summary_io import SummaryWriterCache


def dataset(
    shape: Tuple[int, int],
    batch_size: int = 32,
    epochs: int = None,
    train: bool = False,
    _batch=True,
) -> tf.data.Dataset:
    """Returns the dataset correcly batched and resized
    Args:
        shape: The output shape of the images in the Dataset
        batch_size: The size of the batch to return at each invocation
        epochs: The the number of times that the dataset will repeat itself
                before throwing an exception
        train: when True, returns the shuffled train dataset, when False returns
               the test, not shuffled, dataset
        _batch: private, do not use
    Returns:
        The dataset
    """

    def _process(image, label):
        """The function used to resize the image to the specified shape.
        Used in the tf.data.Dataset.map function
        Args:
            image: the input image
            label: the input label
        Return:
            resized_image, label
        """
        nonlocal shape
        image = tf.image.resize_images(
            tf.expand_dims(image, axis=0), shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        image = tf.cast(image, tf.float32)
        image = tf.squeeze(image, axis=[0])
        return image, label

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = k.datasets.cifar10.load_data()
    if not train:
        train_images = test_images
        train_labels = test_labels
    train_images = (train_images - 127.5) / 127.5

    def _generator():
        r"""The generator that returns the pair image,label
        This must be used in order to don't use tf.data.Dataset.from_tensor_slices.abs
        In this way, we can build a dataset from a generator and solve the problem of huge
        graphs created by from_tensor_slices (it creates a constant in the graph :\)
        """
        for image, label in zip(train_images, train_labels):
            yield image, label

    def _set_shape(image, label):
        """Set the static + dynamic shape of the image, in order to correctly build the
        input pipeline in both phases
        Args:
            image: the input image
            label: the input label
        Return:
            image, label
        """
        image.set_shape((32, 32, 3))  # static
        image = tf.reshape(image, (32, 32, 3))  # dynamic
        return image, label

    _dataset = tf.data.Dataset.from_generator(
        _generator, (tf.float32, tf.int32)
    )  # unkown shape
    _dataset = _dataset.map(
        _set_shape, num_parallel_calls=multiprocessing.cpu_count()
    )  # known static chsape

    _dataset = _dataset.map(
        _process, num_parallel_calls=multiprocessing.cpu_count()
    )  # resize to desired input shape

    if _batch:
        _dataset = _dataset.batch(batch_size, drop_remainder=True)
        if epochs:
            _dataset = _dataset.repeat(epochs)
    elif epochs:
        _dataset = _dataset.repeat(epochs)

    _dataset = _dataset.prefetch(1)
    return _dataset


KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


def discriminator(
    visual_shape: tf.TensorShape,
    encoding_shape: tf.TensorShape,
    conditioning: Optional[Any] = None,
    l2_penalty: float = 0.0,
) -> k.Model:
    """
    Build the Discriminator model.

    Returns a k.Model with 2 inputs and a single output.
    The inputs are an image and its encoded/latent representation.

    Args:
        visual_shape: The shape of the visual input, 3D tensor
        encoding_shape: The shape of the latent/encoded representation
        # NOT IMPLEMENTED: Conditioning: data used as GAN conditioning
        # UNUSED: l2_penalty: l2 regularization strength

    Returns:
        The discriminator model.

    """
    kernel_size = (5, 5)

    # Inputs
    # (64, 64, C)
    # (Latent Dimension, )
    input_visual = k.layers.Input(shape=visual_shape)
    input_encoding = k.layers.Input(shape=encoding_shape)

    # Data
    # ### Layer 0
    # (64, 64, 32)
    visual = k.layers.Conv2D(
        filters=32,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=k.regularizers.l2(l2_penalty),
    )(input_visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)

    # Data
    # ### Layer 1
    # (32, 32, 32)
    visual = k.layers.Conv2D(
        filters=32,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=k.regularizers.l2(l2_penalty),
    )(visual)
    visual = k.layers.BatchNormalization()(visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)
    visual = k.layers.Dropout(rate=0.5)(visual)

    # ### Layer 2
    # (16, 16, 64)
    visual = k.layers.Conv2D(
        filters=64,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
    )(visual)
    visual = k.layers.BatchNormalization()(visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)
    visual = k.layers.Dropout(rate=0.5)(visual)

    # Flatten
    visual = k.layers.Flatten()(visual)

    # Encoding
    # ### Layer 5 D(z)
    # (512,)
    encoding = k.layers.Dense(units=512, kernel_initializer=KERNEL_INITIALIZER)(
        input_encoding
    )
    encoding = k.layers.LeakyReLU(alpha=0.1)(encoding)
    encoding = k.layers.Dropout(rate=0.5)(encoding)

    # Data + Encoding
    # ### Layer 6 D(x, z)
    # (4608,)
    mixed = k.layers.Concatenate()([visual, encoding])
    mixed = k.layers.Dense(units=1024, kernel_initializer=KERNEL_INITIALIZER)(mixed)
    mixed = k.layers.LeakyReLU(alpha=0.1)(mixed)
    mixed = k.layers.Dropout(rate=0.5)(mixed)
    features = mixed

    # Final Step
    # ### Layer 7
    # (1)
    out = k.layers.Dense(1, kernel_initializer=KERNEL_INITIALIZER)(mixed)

    # Use the functional interface
    model = k.Model(inputs=[input_visual, input_encoding], outputs=[out, features])
    model.summary()
    return model


def generator(
    input_shape: int,
    output_depth: int = 3,
    conditioning: Optional[Any] = None,
    l2_penalty: float = 0.0,
) -> k.Model:
    """
    Build the Generator model.

    Given a latent representation, generates a meaningful image.
    The input shape must be in the form of a vector 1x1xD.

    Args:
        input_shape: The shape of the noise prior
        output_depth: The number of channels of the generated image
        # NOT IMPLEMENTED: Conditioning: data used as GAN conditioning
        l2_penalty: l2 regularization strength

    Returns:
        The Generator model.

    """
    kernel_size = (5, 5)
    model = k.Sequential(name="generator")

    # Project and Reshape the latent space
    # ### Layer 1
    # (4*4*64,)
    model.add(
        k.layers.Dense(
            units=4 * 4 * 64,
            kernel_initializer=KERNEL_INITIALIZER,
            input_shape=input_shape,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
        )
    )
    model.add(k.layers.Activation(k.activations.relu))
    model.add(k.layers.Reshape((4, 4, 64)))

    # Starting Deconvolutions
    # ### Layer 2
    # (8, 8, 64)
    model.add(
        k.layers.Conv2DTranspose(
            filters=64,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )
    )
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(k.activations.relu))

    # Starting Deconvolutions
    # ### Layer 3
    # (16, 16, 128)
    model.add(
        k.layers.Conv2DTranspose(
            filters=128,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )
    )
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(k.activations.relu))

    # ### Layer 4
    # (32, 32, 256)
    model.add(
        k.layers.Conv2DTranspose(
            filters=256,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )
    )
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(k.activations.relu))

    # ### Layer 5
    # (64, 64, C)
    model.add(
        k.layers.Conv2DTranspose(
            filters=output_depth,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )
    )
    model.add(k.layers.Activation(k.activations.tanh))  # G(z) in [-1,1]

    model.summary()
    return model


def encoder(
    visual_shape: int, latent_dimension: int, l2_penalty: float = 0.0
) -> k.Model:
    """
    Build the Encoder model.

    The encoder encodes the input in a vector with shape 1x1xlatent_dimension.

    Args:
        visual_shape: The shape of the input to encode
        latent_dimension: The number of dimensions (along the depth) to use.
        # NOT IMPLEMENTED: conditioning: Data used as GAN conditioning
        l2_penalty: l2 regularization strength

    Returns:
        The Encoder model.

    """

    kernel_size = (5, 5)

    # Inputs
    # (64, 64, C)
    # (Latent Dimension, )
    input_visual = k.layers.Input(shape=visual_shape)

    # Data
    # ### Layer 0
    # (64, 64, 32)
    visual = k.layers.Conv2D(
        filters=32,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=k.regularizers.l2(l2_penalty),
    )(input_visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)

    # Data
    # ### Layer 1
    # (32, 32, 32)
    visual = k.layers.Conv2D(
        filters=32,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=k.regularizers.l2(l2_penalty),
    )(visual)
    visual = k.layers.BatchNormalization()(visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)
    visual = k.layers.Dropout(rate=0.5)(visual)

    # ### Layer 2
    # (16, 16, 64)
    visual = k.layers.Conv2D(
        filters=128,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
    )(visual)
    visual = k.layers.BatchNormalization()(visual)
    visual = k.layers.LeakyReLU(alpha=0.1)(visual)
    visual = k.layers.Dropout(rate=0.5)(visual)

    # Flatten
    visual = k.layers.Flatten()(visual)

    # Encoding
    # (Latent space, )
    # ### Layer 5
    visual = k.layers.Dense(
        units=latent_dimension, kernel_initializer=KERNEL_INITIALIZER
    )(visual)

    model = k.Model(inputs=input_visual, outputs=visual)
    model.summary()
    return model


def bce(x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    """Returns the discrete binary cross entropy between x and the discrete label
    Args:
        x: a 2D tensor
        label: the discrite label, aka, the distribution to match

    Returns:
        The binary cros entropy
    """
    assert len(x.shape) == 2 and len(label.shape) == 0

    return tf.losses.sigmoid_cross_entropy(tf.ones_like(x) * label, x)


def min_max(positive: tf.Tensor, negative: tf.Tensor) -> tf.Tensor:
    """Returns the discriminator (min max) loss
    Args:
        positive: the discriminator output for the positive class: 2D tensor
        negative: the discriminator output for the negative class: 2D tensor
    Returns:
        The sum of 2 BCE
    """
    one = tf.constant(1.0)
    zero = tf.constant(0.0)
    d_loss = bce(positive, one) + bce(negative, zero)
    return d_loss


def train():
    """Train the GAN."""
    batch_size = 32
    epochs = 100
    latent_dimension = 100
    l2_penalty = 0.0

    x_, y_ = dataset((64, 64), batch_size, epochs).make_one_shot_iterator().get_next()

    x = tf.placeholder(tf.float32, list(x_.shape))
    tf.summary.image("x", x, max_outputs=3)

    # Define the Models
    E = encoder(x.shape[1:], latent_dimension, l2_penalty)

    z_ = tf.random_normal([batch_size, latent_dimension], mean=0.0, stddev=1.0)

    z = tf.placeholder(tf.float32, list(z_.shape))
    G = generator(z.shape[1:], x.shape[-1].value, l2_penalty)
    D = discriminator(x.shape[1:], E.output.shape[1:], l2_penalty)

    # Generate from latent representation z
    G_z = G(z)
    tf.summary.image("G(z)", G_z, max_outputs=3)

    # encode x to a latent representation
    E_x = E(x)

    G_Ex = G(E_x)
    tf.summary.image("G(E(x))", G_Ex, max_outputs=3)

    # plot image difference
    tf.summary.image(
        "G(E(x)) - x", tf.norm(G_Ex - x_, axis=3, keepdims=True), max_outputs=3
    )

    # The output of the discriminator is a bs,n,n,value
    # hence flatten all the values of the map and compute
    # the loss element wise
    D_Gz, F_Gz = D(inputs=[G_z, z])
    D_x, F_x = D(inputs=[x, E_x])
    D_Gz = k.layers.Flatten()(D_Gz)
    F_Gz = k.layers.Flatten()(F_Gz)
    D_x = k.layers.Flatten()(D_x)
    F_x = k.layers.Flatten()(F_x)

    ## Discriminator
    d_loss = min_max(D_x, D_Gz)

    ## Generator
    g_loss = bce(D_Gz, tf.constant(1.0))
    # Encoder
    e_loss = bce(D_x, tf.constant(0.0))

    # add regularizations defined in the keras layers
    d_loss += tf.add_n(D.losses)
    e_loss += tf.add_n(E.losses)
    g_loss += tf.add_n(G.losses)

    tf.summary.scalar("d_loss", d_loss)
    tf.summary.scalar("g_loss", g_loss)
    tf.summary.scalar("e_loss", e_loss)

    global_step = tf.train.get_or_create_global_step()

    lr = 1e-4
    tf.summary.scalar("lr", lr)

    # Define the D train op
    train_d = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        d_loss, var_list=D.trainable_variables
    )

    # Define the G + E train ops (the models can be trained
    # the same step, but separately)
    train_g = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        g_loss, global_step=global_step, var_list=G.trainable_variables
    )

    train_e = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        e_loss, var_list=E.trainable_variables
    )

    log_dir = f"logs/test"
    summary_op = tf.summary.merge_all()

    scaffold = tf.train.Scaffold()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session_creator = tf.train.ChiefSessionCreator(
        config=config, scaffold=scaffold, checkpoint_dir=log_dir
    )

    def _loop(func: Callable) -> None:
        """
        Execute func for the specified number of epochs or max_steps.

        Args:
            func: callable to loop

        Returns:
            None.
        """
        try:
            while True:
                func()
        except tf.errors.OutOfRangeError:
            pass

    with tf.train.MonitoredSession(
        session_creator=session_creator,
        hooks=[
            tf.train.CheckpointSaverHook(log_dir, save_steps=100, scaffold=scaffold)
            # tf.train.ProfilerHook(save_steps=1000, output_dir=log_dir),
        ],
    ) as sess:
        # Get the summary writer.
        # The rational behind using the writer (from the scaffold)
        # and not using the SummarySaverHook is that we want to log
        # every X steps the output of G, G(E(x)) and x
        # But since we need to use placeholders to feed the same data
        # to G, D and E, we can't use the Hook, because the first
        # sess.run on the data, will trigger the summary save op
        # and the summary save op needs the data from the placeholder
        writer = SummaryWriterCache.get(log_dir)

        def _train():
            # First create the input, that must be shared between the 2
            # training iteration
            real, noise = sess.run([x_, z_])
            feed_dict = {x: real, z: noise}

            # train D
            d_gz_, d_x, _, d_loss_value = sess.run(
                [D_Gz, D_x, train_d, d_loss], feed_dict
            )

            # train G+E
            _, g_loss_value, _, e_loss_value, step = sess.run(
                [train_g, g_loss, train_e, e_loss, global_step], feed_dict
            )

            if step % 100 == 0:
                summaries = sess.run(summary_op, feed_dict)
                print(
                    f"[{step}] d: {d_loss_value} - g: {g_loss_value} - e: {e_loss_value}"
                )
                print(np.mean(d_gz_), np.mean(d_x))
                writer.add_summary(summaries, step)
                writer.flush()

        _loop(_train)
    return 0


if __name__ == "__main__":
    sys.exit(train())

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)