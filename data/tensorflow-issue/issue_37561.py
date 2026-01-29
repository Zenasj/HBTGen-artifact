# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape: batch of images with (height=32, width=128, channels=1) grayscale images typically used in CRNN OCR

import tensorflow as tf

class WordAccuracy(tf.keras.metrics.Metric):
    """
    Calculate word accuracy metric comparing sparse ground truth and predictions.
    This metric decodes y_pred using CTC greedy decoder and compares with y_true.
    """
    def __init__(self, name='word_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        # Use float32 for add_weight to avoid distributed strategy hanging issues
        self.total = self.add_weight(name='total', shape=(), dtype=tf.float32,
                                     initializer='zeros')
        self.count = self.add_weight(name='count', shape=(), dtype=tf.float32,
                                     initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        b = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([b], tf.shape(y_pred)[1])
        
        # CTC greedy decode assumes y_pred shape: (max_time, batch_size, num_classes)
        # y_pred is (batch, max_time, num_classes) - need to transpose for decoder
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        
        # Reset shape of sparse tensors for consistent comparison
        y_true = tf.sparse.reset_shape(y_true, [b, max_width])
        y_pred_sparse = tf.sparse.reset_shape(decoded[0], [b, max_width])
        
        y_true_dense = tf.sparse.to_dense(y_true, default_value=-1)
        y_pred_dense = tf.sparse.to_dense(y_pred_sparse, default_value=-1)
        
        y_true_dense = tf.cast(y_true_dense, tf.int32)
        y_pred_dense = tf.cast(y_pred_dense, tf.int32)
        
        # Compare differences row-wise
        unequal = tf.math.reduce_any(tf.math.not_equal(y_true_dense, y_pred_dense), axis=1)
        unequal_int = tf.cast(unequal, tf.float32)
        num_unequal = tf.reduce_sum(unequal_int)

        self.total.assign_add(tf.cast(b, tf.float32))
        self.count.assign_add(tf.cast(b, tf.float32) - num_unequal)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)

class MyModel(tf.keras.Model):
    """
    Placeholder CRNN model for text recognition.
    Using convolution layers + bidirectional LSTM + dense for CTC output.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        # Assumptions:
        # Input shape: (batch, 32, 128, 1) grayscale images
        # num_classes: number of output classes for CTC (including blank)
        self.conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,1))
        self.conv5 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,1))
        self.reshape = tf.keras.layers.Reshape((-1, 512))  # Time steps depending on width
        # Bidirectional LSTM layers
        self.bi_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.bi_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = inputs  # shape: (batch, 32, 128, 1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pool5(x)
        x = self.reshape(x)  # shape: (batch, time_steps, features)
        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)
        x = self.dense(x)  # logits for CTC: (batch, time_steps, num_classes)
        return x

def my_model_function():
    # Instantiate MyModel and compile with dummy optimizer and loss placeholder
    model = MyModel(num_classes=80)  # Number of classes inferred from CRNN example
    # Using from_logits=True because the model outputs raw logits for CTC
    ctc_loss_fn = tf.keras.backend.ctc_batch_cost

    # Wrapper loss function that matches Keras expectations (y_true sparse tensor)
    def ctc_loss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
        input_length = input_length * tf.ones(shape=(batch_len,1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len,1), dtype='int64')
        loss = ctc_loss_fn(y_true, y_pred, input_length, label_length)
        return tf.reduce_mean(loss)

    # Compile with optimizer, loss, and the custom metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=ctc_loss,
        metrics=[WordAccuracy()]
    )
    return model

def GetInput():
    # Return a random tensor matching expected input shape:
    # batch size 4, height 32, width 128, channels 1 (grayscale)
    # dtype float32 as typical image input normalized
    B = 4
    H = 32
    W = 128
    C = 1
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

