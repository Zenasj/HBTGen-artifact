# tf.random.uniform((B, max_sentence_len, embedding_dim), dtype=tf.float32)
import tensorflow as tf

# Note: Original code is TF 1.x style with placeholders and tf.Session.
# Here is a reconstructed single class MyModel using TF 2.x keras.Model style,
# converting placeholders to input tensors, and reorganizing the logic accordingly.
# For conv layers, we replace `tf.contrib.layers.conv2d` and max_pool2d with Keras layers.
# For missing softmax_layer and transition_layer, simple placeholder dense layers are used,
# with comments referring to their original use.

class MyModel(tf.keras.Model):
    def __init__(self, config,
                 filter_list=(3, 4, 5),
                 filter_num=100,
                 seed=42):
        super().__init__()
        self.config = config
        self.filter_list = filter_list
        self.filter_num = filter_num
        self.seed = seed

        # Embedding weights handled externally (assume initialized outside for now),
        # if config.pre_trained: use constant embeddings, else trainable embeddings
        # For simplicity, initialize embedding layer here with uniform random weights and seed for reproducibility
        if hasattr(config, "vocab_size") and config.vocab_size is not None:
            vocab_size = config.vocab_size
        else:
            vocab_size = 5000  # assumption
        if hasattr(config, "embedding_dim") and config.embedding_dim is not None:
            embedding_dim = config.embedding_dim
        else:
            embedding_dim = 300  # assumption

        # Seed-based initializer for reproducibility of embedding
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=self.seed),
            trainable=(getattr(config, "embedding_type", "non-static") != 'static')
        )

        # CNN conv and max pool layers for each filter size - two sets: clean and noisy CNN
        self.clean_cnn_layers = []
        self.noisy_cnn_layers = []
        for filter_size in filter_list:
            # conv layer: 1D conv with ReLU activation
            conv_layer_clean = tf.keras.layers.Conv2D(
                filters=filter_num,
                kernel_size=(filter_size, embedding_dim),
                strides=(1,1),
                padding='valid',
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
            )
            self.clean_cnn_layers.append(conv_layer_clean)

            conv_layer_noisy = tf.keras.layers.Conv2D(
                filters=filter_num,
                kernel_size=(filter_size, embedding_dim),
                strides=(1,1),
                padding='valid',
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
            )
            self.noisy_cnn_layers.append(conv_layer_noisy)

        # Pooling layers for each filter (pool over time dimension after conv)
        # Implemented in call()

        # Fully connected softmax layer replacing original softmax_layer
        # The original code used a noisy layer and a p1, p2 transition matrices.
        # Here we approximate with dense layers.
        self.fc_softmax = tf.keras.layers.Dense(
            units=getattr(config, "n_class", 2),
            kernel_regularizer=tf.keras.regularizers.l2(getattr(config, "l2_reg", 0.0)),
            activation=None,  # logits, softmax applied in loss
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )

        # Transition layers approximated by Dense layers (p1 and p2), act as learned transitions
        hidden_dim = filter_num * len(filter_list)
        self.p1_layer = tf.keras.layers.Dense(
            units=getattr(config, "n_class", 2),  # output shape consistent with n_class
            kernel_regularizer=tf.keras.regularizers.l2(getattr(config, "l2_reg", 0.0)),
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )
        self.p2_layer = tf.keras.layers.Dense(
            units=getattr(config, "n_class", 2),
            kernel_regularizer=tf.keras.regularizers.l2(getattr(config, "l2_reg", 0.0)),
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate=1-getattr(config, "keep_prob1", 1.0), seed=self.seed)
        self.dropout2 = tf.keras.layers.Dropout(rate=1-getattr(config, "keep_prob2", 1.0), seed=self.seed)

    def _apply_cnn_layers(self, inputs, cnn_layers):
        # inputs: shape (B, max_sentence_len, embedding_dim)
        # Expand dim for conv2d to 4D: (B, H, W, C) with W=embedding_dim and C=1 channel
        x = tf.expand_dims(inputs, axis=-1)  # (B, max_len, embedding_dim, 1)
        pooled_outputs = []
        max_len = tf.shape(inputs)[1]
        embedding_dim = tf.shape(inputs)[2]
        for i, conv_layer in enumerate(cnn_layers):
            conv = conv_layer(x)  # shape (B, max_len - filter_size + 1, 1, filter_num)
            # max pooling over the time dimension (height axis = conv.shape[1])
            pool_size = conv.shape[1]
            pool = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,1), strides=(1,1), padding='valid')(conv)  # shape (B,1,1,filter_num)
            pooled_outputs.append(pool)
        # Concatenate pooled outputs over filters along last axis
        hiddens = tf.concat(pooled_outputs, axis=-1)  # (B,1,1, filter_num * len(filter_list))
        hiddens = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])  # (B, hidden_dim)
        return hiddens

    def call(self, inputs, training=False):
        """
        inputs: a dict or tuple containing:
            'sen_x_batch': int tensor shape (B, max_sentence_len) -- word ids
        or directly a tensor (B, max_sentence_len) of word indices (integers)
        
        Returns:
          noisy_logits: logits after noisy adaptation (B, n_class)
          clean_logits: clean logits directly from CNN softmax (B, n_class)
        """
        if isinstance(inputs, dict):
            sen_x_batch = inputs.get('sen_x_batch')
        else:
            sen_x_batch = inputs

        # Embed words (B, max_sentence_len, embedding_dim)
        x_embed = self.embedding_layer(sen_x_batch)  

        # Apply first dropout (like keep_prob1)
        x_dropout = self.dropout1(x_embed, training=training)

        # CNN layers produce clean features and noisy features
        clean_features = self._apply_cnn_layers(x_dropout, self.clean_cnn_layers)
        noisy_features = self._apply_cnn_layers(x_dropout, self.noisy_cnn_layers)

        # Fully connected layer for clean logits
        clean_logits = self.fc_softmax(self.dropout2(clean_features, training=training))  # shape (B, n_class)

        # Transition matrices p1 and p2 from noisy features
        p1 = self.p1_layer(noisy_features)  # (B, n_class)
        p2 = self.p2_layer(noisy_features)  # (B, n_class)

        p1_exp = tf.expand_dims(p1, 2)  # (B, n_class, 1)
        p2_exp = tf.expand_dims(p2, 2)  # (B, n_class, 1)
        prob = tf.concat([p1_exp, p2_exp], axis=2)  # (B, n_class, 2) [as original]

        sen_logits_exp = tf.expand_dims(clean_logits, 1)  # (B,1,n_class)

        # noisy_logits = tf.squeeze(tf.matmul(sen_logits_exp, prob)) replicates
        # matmul shape: (B,1,n_class) x (B,n_class,2) => (B,1,2) squeeze->(B,2)
        noisy_logits = tf.matmul(sen_logits_exp, prob)  # (B,1,2)
        noisy_logits = tf.squeeze(noisy_logits, axis=1)  # (B,2)

        return noisy_logits, clean_logits


def my_model_function():
    # Dummy config class with minimal needed attributes for instantiation
    class Config:
        max_sentence_len = 50
        embedding_dim = 300
        n_class = 2
        keep_prob1 = 0.8
        keep_prob2 = 0.5
        l2_reg = 0.001
        embedding_type = 'non-static'  # trainable embeddings
        random_base = 0.1
        # fill vocab_size for embedding init
        vocab_size = 5000
    config = Config()
    return MyModel(config=config)


def GetInput():
    """
    Returns random valid input tensor matching the expected input:
    A batch of integer IDs for words of shape [B, max_sentence_len]
    """
    B = 16  # batch size assumption
    max_sentence_len = 50  # from Config default
    vocab_size = 5000

    import numpy as np

    # generate random integer word ids between 0 and vocab_size-1
    input_tensor = tf.constant(
        np.random.randint(low=0, high=vocab_size, size=(B, max_sentence_len)), dtype=tf.int32
    )
    return input_tensor

