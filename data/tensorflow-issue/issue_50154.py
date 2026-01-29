# tf.random.uniform((7, 50, 100), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dropout, TimeDistributed, Dense, Lambda
from tensorflow_addons.metrics import F1Score
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.masking = Masking()
        self.bilstm = Bidirectional(
            LSTM(
                units=16,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
            )
        )
        self.dropout = Dropout(rate=0.5, seed=42)
        self.time_dist_dense = TimeDistributed(Dense(5, activation="softmax"), name="logits")
        self.naming_layer = Lambda(lambda x: x, name="pred_ids")
        # F1Score metric for logits output:
        self.f1score_metric = F1Score(num_classes=5, average="micro")

    def call(self, inputs, training=False):
        # inputs is a tuple/dict of (embeddings, nwords)
        embeddings = inputs["embedding_sequence"] if isinstance(inputs, dict) else inputs[0]
        nwords = inputs["nwords"] if isinstance(inputs, dict) else inputs[1]

        masked_embedding = self.masking(embeddings)
        bilstm_output = self.bilstm(masked_embedding, training=training)
        bilstm_output = self.dropout(bilstm_output, training=training)
        logits = self.time_dist_dense(bilstm_output)
        pred_ids = tf.argmax(logits, axis=2, output_type=tf.int32)
        pred_ids = self.naming_layer(pred_ids)
        # Return a dict matching model output names
        return {"logits": logits, "pred_ids": pred_ids}

    # We override train_step to handle the mask/sample_weight issue:
    def train_step(self, data):
        # Unpack the data. Its structure depends on the model's inputs and outputs
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            # Only apply categorical_crossentropy to logits (not pred_ids), no mask/sample_weight issue here
            loss = self.compiled_loss(y["logits"], y_pred["logits"], regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics - workaround mask/sample_weight type issue:
        # We do NOT pass the mask (boolean) as sample_weight which was causing the TypeError
        for metric in self.metrics:
            if isinstance(metric, F1Score):
                # Update metric directly with y_true and y_pred (logits) without sample_weight
                metric.update_state(y["logits"], y_pred["logits"])
            else:
                metric.update_state(y["logits"], y_pred["logits"])

        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    # Instantiate and compile the model matching original example's compile
    model = MyModel()
    # Compile with Adam optimizer and categorical_crossentropy loss on logits output only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"logits": "categorical_crossentropy"},
        # Metrics only on logits output, F1Score metric is handled internally in train_step
        metrics=[],
        run_eagerly=False
    )
    return model

def GetInput():
    window_length = 50
    embedding_dimension = 100
    batch_size = 7
    # Create dummy inputs with correct shapes and dtypes matching original example
    embeddings = tf.random.uniform((batch_size, window_length, embedding_dimension), dtype=tf.float32)
    nwords = tf.constant([window_length]*batch_size, dtype=tf.int32)

    # Create dummy labels matching model outputs:
    # logits shape: (batch_size, window_length, num_classes=5), use zeros
    logits = tf.zeros((batch_size, window_length, 5), dtype=tf.float32)
    # pred_ids is expected but not used in loss, can provide zeros as well
    pred_ids = tf.zeros((batch_size, window_length), dtype=tf.int32)

    input_dict = {"embedding_sequence": embeddings, "nwords": nwords}
    label_dict = {"logits": logits, "pred_ids": pred_ids}
    return (input_dict, label_dict)

