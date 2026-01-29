# tf.random.uniform((BATCH_SIZE, SEQ_LEN + OVERLAP, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

NUM_EPOCHS = 2
BATCH_SIZE = 16
FRAME_SIZE = 64
NUM_BATCHES = 4
SEQ_LEN = 1024
OVERLAP = FRAME_SIZE
NUM_SAMPS = 128000 + OVERLAP

def pad_batch(batch, batch_size, seq_len, overlap):
    # Pads each batch with zeros for overlap amount on the left
    # and trims to a multiple of seq_len.
    num_samps = (int(np.floor(len(batch[0]) / float(seq_len))) * seq_len)
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    # batch: shape = (batch_size, num_samps_total, 1)
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def get_subseqs_from_batch(batch, seq_len=SEQ_LEN, overlap=OVERLAP):
    # Generator yielding subsequences with overlap from a batch.
    # batch is a tf.Tensor shaped (batch_size, time, 1)
    batch_np = batch.numpy()
    batch_size = batch_np.shape[0]
    num_samps = batch_np.shape[1]
    for i in range(overlap, num_samps, seq_len):
        x = batch_np[:, i - overlap:i + seq_len]       # (batch_size, seq_len + overlap, 1)
        y = x[:, overlap:overlap + seq_len]            # (batch_size, seq_len, 1)
        yield (x, y)

def get_dataset(files=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, overlap=OVERLAP,
                seq_len=SEQ_LEN, drop_remainder=False):
    # Dataset that loads audio batches and yields subsequences.
    # This version uses a single from_generator for loading audio,
    # then batch, padding, and then a map producing subsequences via generator iteration.
    # This avoids the second from_generator which caused hangs in some environments.
    def load_audio(files, batch_size):
        # Dummy audio loader simulating batches of audio samples.
        for filename in range(350):
            audio = np.random.randint(0, 256, size=(batch_size * NUM_SAMPS))
            audio = audio.reshape((batch_size, NUM_SAMPS, 1)).astype('float32')
            # Yield one batch at a time
            yield audio

    base_dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, batch_size),
        output_types=tf.float32,
        output_shapes=(batch_size, None, 1)
    )
    base_dataset = base_dataset.repeat(num_epochs).batch(1, drop_remainder=drop_remainder)
    # Because load_audio yields batches of shape (batch_size, time, 1),
    # batching again with size 1 results in (1, batch_size, time, 1), so we can squeeze later.
    base_dataset = base_dataset.map(lambda x: tf.squeeze(x, axis=0))  # shape (batch_size, time, 1)
    
    # pad batches
    def pad_fn(batch):
        return tf.py_function(func=pad_batch,
                              inp=[batch, batch_size, seq_len, overlap],
                              Tout=tf.float32)
    dataset = base_dataset.map(pad_fn)

    # Wrap the generator that yields subsequences from each batch as a Dataset
    def subseq_gen():
        for batch in dataset:
            for x, y in get_subseqs_from_batch(batch, seq_len=seq_len, overlap=overlap):
                yield tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)

    output_types = (tf.float32, tf.float32)
    output_shapes = ((batch_size, seq_len + overlap, 1), (batch_size, seq_len, 1))
    final_dataset = tf.data.Dataset.from_generator(subseq_gen,
                                                   output_types=output_types,
                                                   output_shapes=output_shapes)
    return final_dataset

class MyModel(tf.keras.Model):

    def __init__(self, frame_size=FRAME_SIZE, dim=1024):
        super(MyModel, self).__init__()
        self.frame_size = frame_size
        self.q_levels = 256
        self.dim = dim
        self.num_lower_tier_frames = 4

        self.input_expand = tf.keras.layers.Dense(self.dim)
        self.rnn = tf.keras.layers.GRU(self.dim, return_sequences=True, stateful=True)
        # Upsample variable: shape [num_lower_tier_frames, dim, dim]
        self.upsample = tf.Variable(
            tf.keras.initializers.GlorotNormal()(
                shape=[self.num_lower_tier_frames, self.dim, self.dim]),
            trainable=True,
            name="upsample",
        )
        self.out = tf.keras.layers.Dense(self.q_levels, activation='relu')

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: Tensor of shape (batch_size, time, 1), float32
        """
        batch_size = tf.shape(inputs)[0]
        # Cast inputs to float32 for safe processing
        x = tf.cast(inputs[:, :-self.frame_size, :], tf.float32)  # trim last frame_size samples

        # Reshape to frames
        frames = tf.reshape(x, [
            batch_size,
            tf.shape(x)[1] // self.frame_size,
            self.frame_size
        ])  # shape (batch, num_frames, frame_size)

        # Pass frames through dense layer
        frames_expanded = self.input_expand(frames)  # (batch, num_frames, dim)

        # Pass through stateful GRU
        hidden = self.rnn(frames_expanded, training=training)  # (batch, num_frames, dim)

        num_steps = tf.shape(frames)[1]

        # Compute output shape for conv1d_transpose (upsampling)
        output_shape = [
            batch_size,
            num_steps * self.num_lower_tier_frames,
            self.dim
        ]

        # Conv1D transpose (upsampling) over hidden states with learned upsample variable
        outputs = tf.nn.conv1d_transpose(
            hidden,
            self.upsample,
            output_shape=output_shape,
            strides=self.num_lower_tier_frames,
            padding='VALID'
        )  # shape (batch_size, output_length, dim)

        # Transpose to (batch, dim, time) then Dense -> q_levels with relu activation
        outputs_transposed = tf.transpose(outputs, perm=(0, 2, 1))  # (batch, dim, time)

        out = self.out(outputs_transposed)  # (batch, q_levels, time)
        return out  # Return logits or features for further processing

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input:
    # shape: (BATCH_SIZE, SEQ_LEN + OVERLAP, 1), dtype=float32
    # Generates uniform random float values between 0 and 255
    return tf.random.uniform(
        shape=(BATCH_SIZE, SEQ_LEN + OVERLAP, 1),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )

