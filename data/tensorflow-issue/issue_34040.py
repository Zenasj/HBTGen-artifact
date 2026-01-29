# tf.random.uniform((B, 512, 300), dtype=tf.float32)

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters as in the issue
        self.window_length = 512
        self.embedding_dimension = 300
        self.lstm_size = 10
        
        # Tag set - fixed order for indexing consistency
        self.tags = ['I-tag1', 'B-tag1', 'I-tag2', 'B-tag2', 'I-tag3', 'B-tag3', 'O']
        self.num_tags = len(self.tags)
        
        self.dropout_rate = 0.5
        
        # Layers
        self.lstm = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(self.num_tags, name="myscores")
        
        # Precompute transition params tensor for CRF based on IOB tag constraints
        self.transition_params = self._compute_transition_params()
        
    def _split_iob_tag(self, tag):
        # Helper from issue to split tags into type and name
        if not (tag.startswith("B-") or tag.startswith("I-")):
            return (tag, None)
        return tuple(tag.split("-", maxsplit=1))
    
    def _compute_transition_params(self):
        # Compute mask matrix with -inf for illegal transitions as per _split_iob_tag logic
        mat = np.zeros((self.num_tags, self.num_tags), dtype=np.float32)
        
        for i, i_tag in enumerate(self.tags):
            i_type, i_name = self._split_iob_tag(i_tag)
            for j, j_tag in enumerate(self.tags):
                j_type, j_name = self._split_iob_tag(j_tag)
                # Disallow transitions:
                # O to I-*, or B/I-* to I-* with different entity type
                if (i_type == "O" and j_type == "I") or \
                   (i_type in ["B", "I"] and j_type == "I" and i_name != j_name):
                    mat[i][j] = -np.inf
        return tf.constant(mat)
    
    def call(self, inputs, training=False):
        # inputs: tuple or list of (embeddings, nwords)
        embeddings, nwords = inputs
        x = self.lstm(embeddings)
        x = self.dropout(x, training=training)
        logits = self.dense(x)  # shape (batch, time, num_tags)
        return logits
    
    def get_custom_loss(self):
        # Return a loss function that captures the CRF log likelihood using sequence lengths from nwords input
        # This loss expects y_true: shape [batch, time], integer tag indices
        transition_params = self.transition_params

        def CRFLoss(seqlen):
            def loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.int32)
                # Ensure shape is [batch, time]
                y_true = tf.reshape(y_true, [-1, self.window_length])
                # y_pred is logits with shape [batch, time, num_tags]
                log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
                    inputs=y_pred,
                    tag_indices=y_true,
                    sequence_lengths=seqlen,
                    transition_params=transition_params,
                )
                return tf.reduce_mean(-log_likelihood)
            return loss

        return CRFLoss



def my_model_function():
    # Instantiate MyModel and compile with custom loss that uses nwords input
    model = MyModel()
    # The model expects two inputs — embeddings and nwords.
    # We'll compile with optimizer Adam and the custom loss bound to nwords input.
    # The custom loss function depends on nwords tensor during training.
    # We wrap the loss to accept y_true and y_pred only, but internally get nwords from input.
    # Because the custom loss requires the dynamic sequence length (nwords), 
    # some workaround is typically required — but we replicate the user logic as closely as possible here.
    
    # For keras Model compile, loss functions must accept y_true,y_pred,
    # but here we need nwords (sequence length) as well.
    # To solve this in a tf.keras.Model subclass, loss must close over nwords input tensor.
    # We assume user supplies y_true with appropriate shape.
    
    # Define a loss function factory that takes nwords tensor and returns a callable loss function
    # We'll create a dummy loss that requires the nwords input to be passed during training and get it from the inputs tensor.
    
    # Since Keras loss functions don't allow extra inputs by default, typically handled by subclassed training loops.
    # For the sake of the converted model, we create a wrapper loss provided nwords tensor is accessible.
    # Here, just demonstrate loss factory usage for compatibility.
    
    # We create a helper Keras layer that feeds nwords to loss function via model's call.
    # But for simplicity in this code, compile with a placeholder loss,
    # as passing nwords dynamically to loss in standard compile is not straightforward.
    
    # Instead, provide loss such that in a custom training loop it can be used with nwords.
    # For demonstration, we bind nwords input tensor dynamically via a tf.function to simulate original behavior.
    
    # We'll compile with a placeholder loss returning zeros (to avoid error on compile).
    # Users must override training step or use custom training loop to get correct training with this loss.
    
    # (The original issue relates to model_to_estimator not supporting stateful custom loss with extra inputs,
    # which is precisely the problem; here we replicate model structure)
    
    # Using a dummy loss here for code completeness:
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)),
                  metrics=['accuracy'])
    
    return model


def GetInput():
    # Return a tuple of inputs matching model inputs:
    # embeddings: tensor with shape (batch_size, window_length, embedding_dimension)
    # nwords: tensor with shape (batch_size,), dtype int32 with sequence lengths <= window_length
    
    batch_size = 3  # from issue params
    window_length = 512
    embedding_dimension = 300
    
    embeddings = tf.random.uniform(
        shape=(batch_size, window_length, embedding_dimension),
        dtype=tf.float32
    )
    
    # nwords is sequence length per sample, here random int <= window_length
    # Generate random sequence lengths between 1 and window_length
    nwords = tf.random.uniform(
        shape=(batch_size,),
        minval=1,
        maxval=window_length + 1,
        dtype=tf.int32
    )
    return (embeddings, nwords)

