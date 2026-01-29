# tf.random.uniform((batch_size, MAX_QUERY_WORD_LENGTH), dtype=tf.int32), tf.random.uniform((batch_size, MAX_TITLE_WORD_LENGTH), dtype=tf.int32)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Assumptions:
# - Input shapes for query and title are (MAX_QUERY_WORD_LENGTH,) and (MAX_TITLE_WORD_LENGTH,)
#   respectively, representing sequences of token ids (integers).
# - Use fixed constants for maximum lengths (could be replaced with actual values).
# - This model fuses the embedding model and custom loss logic into one Keras Model subclass.
# - We include the HardNegativeContrastiveLoss as a tf.keras.losses.Loss subclass,
#   made compatible for saving/loading with get_config.
# - Since the original issue is about serialization problems of custom Loss class,
#   this design allows integrating loss computation as an output layer named 'loss_output'.
# - The model outputs both 'loss_output' and a similarity score 'similarity_output'.
# - The forward call outputs a dict of these two outputs.
# - GetInput() returns a tuple of random integer tensors matching input shapes.

# Placeholder constants (should be set to actual values in real use)
MAX_QUERY_WORD_LENGTH = 10
MAX_TITLE_WORD_LENGTH = 12
BATCH_SIZE = 4
EMBEDDING_DIM = 128

class HardNegativeContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name='hard_negative_contrastive_loss'):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true is unused in original logic
        # y_pred has shape: (batch_size, embedding_dim, 2)
        embedding1 = y_pred[:, :, 0]
        embedding2 = y_pred[:, :, 1]

        embedding1_norm = K.l2_normalize(embedding1, axis=1)
        embedding1_norm_t = K.transpose(embedding1_norm)

        embedding2_norm = K.l2_normalize(embedding2, axis=1)

        dot_product = K.dot(embedding2_norm, embedding1_norm_t)
        distance = 2.0 - 2.0 * dot_product

        mask = tf.cast(tf.equal(distance, 0.0), 'float32')
        distance = distance + mask * 1e-16
        distance = tf.sqrt(distance)
        distance = distance * (1.0 - mask)

        positive_score = tf.linalg.diag_part(distance)

        num_sample = tf.shape(embedding1)[0]
        indices_equal = tf.cast(tf.eye(num_sample), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        mask_indices_not_equal = tf.cast(indices_not_equal, 'float32')

        negative_scores = tf.multiply(mask_indices_not_equal, distance)

        hardest_negative_score_embedding1 = tf.reduce_min(
            negative_scores + 4.0 * tf.cast(indices_equal, 'float32'), axis=1)
        hardest_negative_score_embedding2 = tf.reduce_min(
            negative_scores + 4.0 * tf.cast(indices_equal, 'float32'), axis=0)

        score = K.mean(2.0 * positive_score +
                       tf.maximum(self.margin - hardest_negative_score_embedding1, 0) +
                       tf.maximum(self.margin - hardest_negative_score_embedding2, 0))
        return score

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "margin": self.margin}

class MyModel(tf.keras.Model):
    def __init__(self,
                 query_vocab_size,
                 title_vocab_size,
                 embedding_dim=EMBEDDING_DIM,
                 use_mask_zero=True,
                 margin=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        # Embedding layers for query and title
        self.query_embedding = layers.Embedding(
            input_dim=query_vocab_size + 1,
            output_dim=embedding_dim,
            mask_zero=use_mask_zero,
            name="query_embedding")
        self.query_batch_norm = layers.BatchNormalization(name='query_embedding_normalized')
        self.query_mean = layers.GlobalAveragePooling1D(name='query_mean')

        self.title_embedding = layers.Embedding(
            input_dim=title_vocab_size + 1,
            output_dim=embedding_dim,
            mask_zero=use_mask_zero,
            name="title_embedding")
        self.title_batch_norm = layers.BatchNormalization(name='title_embedding_normalized')
        self.title_mean = layers.GlobalAveragePooling1D(name='title_mean')

        self.concat_for_loss = layers.Concatenate(axis=-1, name='concat_for_loss')
        self.margin = margin
        self.loss_layer_name = 'loss_output'

        # Output layer to produce tensor shaped (batch_size, embedding_dim, 2)
        # by stacking embeddings along last dimension for loss calculation
        # Use Lambda layers for expand_dims and concatenation
        self.expand_dims_query = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.expand_dims_title = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.concat_embeddings = layers.Concatenate(axis=-1, name='loss_output')

        # Similarity output using dot product with normalization
        self.similarity_output = layers.Dot(axes=1, normalize=True, name='similarity_output')

        # Instantiate custom loss instance for access if needed
        self.custom_loss = HardNegativeContrastiveLoss(margin=self.margin)

        # Optimizer (Adam default params)
        self.optimizer = Adam()
        self.compile(
            optimizer=self.optimizer,
            loss={self.loss_layer_name: self.custom_loss},
            metrics={ 'similarity_output': 'binary_accuracy' })

    def call(self, inputs, training=False):
        # inputs is a tuple/list of two tensors: (query_input, title_input)
        query_input, title_input = inputs

        # Compute query embedding vector
        q_emb = self.query_embedding(query_input)  # (batch, query_len, emb_dim)
        q_emb_norm = self.query_batch_norm(q_emb, training=training)
        q_vec = self.query_mean(q_emb_norm)  # (batch, emb_dim)

        # Compute title embedding vector
        t_emb = self.title_embedding(title_input)  # (batch, title_len, emb_dim)
        t_emb_norm = self.title_batch_norm(t_emb, training=training)
        t_vec = self.title_mean(t_emb_norm)  # (batch, emb_dim)

        # Create loss_output tensor by stacking embeddings along last dim
        q_vec_exp = self.expand_dims_query(q_vec)  # (batch, emb_dim, 1)
        t_vec_exp = self.expand_dims_title(t_vec)  # (batch, emb_dim, 1)
        loss_output = self.concat_embeddings([q_vec_exp, t_vec_exp])  # (batch, emb_dim, 2)

        # Similarity output: cosine similarity between q_vec and t_vec
        sim_output = self.similarity_output([t_vec, q_vec])  # (batch, 1)

        return {self.loss_layer_name: loss_output, 'similarity_output': sim_output}

def my_model_function():
    # For demonstration, use some example vocab sizes
    query_vocab_size = 1000
    title_vocab_size = 2000
    return MyModel(query_vocab_size=query_vocab_size,
                   title_vocab_size=title_vocab_size,
                   embedding_dim=EMBEDDING_DIM,
                   use_mask_zero=True,
                   margin=1.0)

def GetInput():
    # Return a tuple of int32 tensors representing query and title input token IDs
    # Here batch size is BATCH_SIZE, with fixed lengths
    query_input = tf.random.uniform(
        shape=(BATCH_SIZE, MAX_QUERY_WORD_LENGTH), minval=1, maxval=1000, dtype=tf.int32)
    title_input = tf.random.uniform(
        shape=(BATCH_SIZE, MAX_TITLE_WORD_LENGTH), minval=1, maxval=2000, dtype=tf.int32)

    return (query_input, title_input)

