# tf.random.uniform((1, 10), dtype=tf.int32), tf.random.uniform((1, 10), dtype=tf.bool), tf.random.uniform((1,), dtype=tf.int32)
import numpy as np
import tensorflow as tf


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return x * tf.sigmoid(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation('linear'),
    "tanh": tf.keras.layers.Activation('tanh'),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish)
}


class TFFastSpeechEmbeddings(tf.keras.layers.Layer):
    """Construct character/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        # Use fixed sin/cos positional embeddings, not trainable
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False,
        )

        if config.n_speakers > 1:
            self.encoder_speaker_embeddings = tf.keras.layers.Embedding(
                config.n_speakers,
                config.hidden_size,
                embeddings_initializer=get_initializer(self.initializer_range),
                name="speaker_embeddings"
            )
            self.speaker_fc = tf.keras.layers.Dense(units=config.hidden_size, name='speaker_fc')
        else:
            self.encoder_speaker_embeddings = None
            self.speaker_fc = None

    def build(self, input_shape):
        """Build shared character/phoneme embedding weights."""
        with tf.name_scope("character_embeddings"):
            self.charactor_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Get character embeddings of inputs.

        Args:
            inputs: tuple/list of (input_ids: int32 tensor shape [batch_size, length],
                                   speaker_ids: int32 tensor shape [batch_size])

        Returns:
            float tensor shape [batch_size, length, hidden_size]
        """
        input_ids, speaker_ids = inputs
        input_shape = tf.shape(input_ids)
        seq_length = input_shape[1]

        position_ids = tf.range(1, seq_length + 1, dtype=tf.int32)[tf.newaxis, :]

        inputs_embeds = tf.gather(self.charactor_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        if self.config.n_speakers > 1 and self.encoder_speaker_embeddings is not None:
            speaker_embeddings = self.encoder_speaker_embeddings(speaker_ids)
            speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))
            # extended speaker embeddings broadcast along sequence length dim
            extended_speaker_features = speaker_features[:, tf.newaxis, :]
            embeddings += extended_speaker_features

        return embeddings

    def _sincos_embedding(self):
        """Create sinusoidal positional encoding."""
        position_enc = np.array([
            [pos / np.power(10000, 2.0 * (i // 2) / self.hidden_size) for i in range(self.hidden_size)]
            for pos in range(self.config.max_position_embeddings + 1)
        ])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc[0] = 0.0  # pad embedding is zero vector
        return position_enc.astype(np.float32)


class TFFastSpeechLengthRegulator(tf.keras.layers.Layer):
    """
    FastSpeech length regulator module that repeats encoder hidden states
    according to durations to expand sequence.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        """
        Args:
            inputs: tuple of (encoder_hidden_states, durations_gt)
                encoder_hidden_states: float32 tensor [batch_size, length, hidden_size]
                durations_gt: int32 tensor [batch_size, length] (rounded durations)
        Returns:
            encoder_masks: int32 tensor [batch_size, max_duration_sum]
              mask with 1 for valid timesteps and 0 padded positions
        """
        encoder_hidden_states, durations_gt = inputs
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)  # scalar

        input_shape = tf.shape(encoder_hidden_states)
        batch_size = input_shape[0]
        hidden_size = input_shape[2]

        # initialize outputs and masks as empty tensors
        outputs = tf.zeros(shape=[0, max_durations, hidden_size], dtype=tf.float32)
        encoder_masks = tf.zeros(shape=[0, max_durations], dtype=tf.int32)

        def condition(i,
                      batch_size,
                      outputs,
                      encoder_masks,
                      encoder_hidden_states,
                      durations_gt,
                      max_durations):
            return tf.less(i, batch_size)

        def body(i,
                 batch_size,
                 outputs,
                 encoder_masks,
                 encoder_hidden_states,
                 durations_gt,
                 max_durations):
            repeats = durations_gt[i]  # [length]
            real_length = tf.reduce_sum(repeats)  # scalar
            pad_size = max_durations - real_length
            masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)  # [1, max_durations]

            # Repeat encoder_hidden_states[i] rows by repeats counts
            repeated_hidden_states = tf.repeat(
                encoder_hidden_states[i],
                repeats=repeats,
                axis=0
            )  # shape [real_length, hidden_size]
            # Pad to max_durations length:
            repeated_hidden_states = tf.pad(repeated_hidden_states, [[0, pad_size], [0, 0]])
            repeated_hidden_states = tf.expand_dims(repeated_hidden_states, 0)  # [1, max_durations, hidden_size]

            outputs = tf.concat([outputs, repeated_hidden_states], axis=0)
            encoder_masks = tf.concat([encoder_masks, masks], axis=0)  # [batch_so_far + 1, max_durations]

            return [i + 1, batch_size, outputs, encoder_masks, encoder_hidden_states, durations_gt, max_durations]

        i = tf.constant(0, dtype=tf.int32)
        _, _, outputs, encoder_masks, _, _, _ = tf.while_loop(
            condition,
            body,
            [i, batch_size, outputs, encoder_masks, encoder_hidden_states, durations_gt, max_durations],
            shape_invariants=[
                i.get_shape(),
                batch_size.get_shape(),
                tf.TensorShape([None, None, self.config.hidden_size]),
                tf.TensorShape([None, None]),
                encoder_hidden_states.get_shape(),
                durations_gt.get_shape(),
                max_durations.get_shape()]
        )

        # For compatibility with TFLite, only return encoder_masks (simplified).
        # Normally outputs is important, but TFLite failures are due to dynamic resizing outputs.
        return encoder_masks


class MyModel(tf.keras.Model):
    """
    A FastSpeech model fusion encapsulating embeddings, encoder, duration predictor,
    length regulator, decoder, mel_dense, and postnet modules.

    This model uses fixed input shapes:
    - input_ids: [batch=1, seq_len=10], tf.int32
    - attention_mask: [batch=1, seq_len=10], tf.bool
    - speaker_ids: [batch=1], tf.int32
    """

    def __init__(self):
        super().__init__()
        # We define a minimal config to match the inputs and sizes inferred from the issue and source
        class DummyConfig:
            vocab_size = 100  # placeholder vocab size
            hidden_size = 384
            initializer_range = 0.02
            max_position_embeddings = 512
            n_speakers = 1
            num_attention_heads = 2
            output_attentions = False
            output_hidden_states = False
            layer_norm_eps = 1e-6
            hidden_dropout_prob = 0.1
            attention_probs_dropout_prob = 0.1
            intermediate_size = 768
            intermediate_kernel_size = 3
            num_hidden_layers = 2
            n_conv_postnet = 5
            postnet_conv_filters = 256
            postnet_conv_kernel_sizes = 5
            postnet_dropout_rate = 0.5
            num_mels = 80
            num_duration_conv_layers = 2
            duration_predictor_filters = 256
            duration_predictor_kernel_sizes = 3
            duration_predictor_dropout_probs = 0.1

        self.config = DummyConfig()

        self.embeddings = TFFastSpeechEmbeddings(self.config, name='embeddings')
        self.length_regulator = TFFastSpeechLengthRegulator(self.config, name='length_regulator')

        # We will implement a simplified forward that uses only embeddings and length regulator.
        # This simplified model captures the part suspected to cause TFLite error.
        # The forward call returns the encoder mask from length regulator.

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 10], dtype=tf.int32),
        tf.TensorSpec(shape=[1, 10], dtype=tf.bool),
        tf.TensorSpec(shape=[1], dtype=tf.int32)
    ])
    def call(self, input_ids, attention_mask, speaker_ids, training=False):
        """
        Forward pass:
        input_ids: int32 [1,10]
        attention_mask: bool [1,10]
        speaker_ids: int32 [1]
        Returns encoder_masks from the length regulator.
        """
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        # For simplicity, make durations all ones (repeat 1 time) to create valid durations tensor.
        # Shape [batch=1, length=10]
        durations = tf.ones(shape=[1, 10], dtype=tf.int32)

        encoder_masks = self.length_regulator([embedding_output, durations], training=training)

        # Return the encoder masks tensor indicating valid frames post length regulation
        return encoder_masks


def my_model_function():
    """
    Returns an instance of MyModel
    """
    return MyModel()


def GetInput():
    """
    Returns input tuple matching MyModel input signature:
    - input_ids: tf.int32 tensor of shape [1,10], with values in [1..vocab_size-1]
    - attention_mask: tf.bool tensor of shape [1,10], all True
    - speaker_ids: tf.int32 tensor of shape [1], value 0
    """
    input_ids = tf.random.uniform(
        shape=(1, 10),
        minval=1,
        maxval=100,
        dtype=tf.int32
    )
    attention_mask = tf.ones(shape=(1, 10), dtype=tf.bool)  # All True mask
    speaker_ids = tf.zeros(shape=(1,), dtype=tf.int32)  # single speaker 0

    return (input_ids, attention_mask, speaker_ids)

