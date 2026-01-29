# tf.random.uniform((B, timestamp-5, vec_length), dtype=tf.float32) ← Input shape derived from Tem_Agg.build_model input: (timestamp-5, vec_length)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, Add
from tensorflow.keras.losses import MeanAbsolutePercentageError

class MyModel(tf.keras.Model):
    def __init__(self, timestamp=300, vec_length=128, model_type="dynamic"):
        """
        Combined model implementing:
        - Temporal aggregation autoencoder architecture (baseline or dynamic)
        - FiLM modulation if dynamic mode is used
        - Custom training step implementing weighted multi-task losses
        """
        super().__init__()
        assert model_type in ["baseline", "dynamic"]
        self.timestamp = timestamp
        self.vec_length = vec_length
        self.model_type = model_type
        
        # Define input shapes:
        # Input shape is (timestamp-5, vec_length), weights shape is (2,)
        self.input_shape_ = (self.timestamp - 5, self.vec_length)

        # Baseline and dynamic share most layers, FiLM applied only if dynamic
        # Encoder layers
        self.enc_lstm1 = LSTM(self.vec_length // 2, return_sequences=True)
        self.enc_lstm2 = LSTM(self.vec_length // 4, return_sequences=True)
        self.enc_lstm_full = LSTM(self.vec_length, name="representation")

        # Decoder layers for reconstruction
        self.dec_lstm1 = LSTM(self.vec_length // 2, return_sequences=True)
        self.dec_lstm_recon = LSTM(self.vec_length, return_sequences=True)

        # Dense layers for future prediction
        self.pred_dense1 = Dense(self.vec_length, activation="relu")
        self.pred_dense2 = Dense(self.vec_length, activation="relu")
        self.pred_dense3 = Dense(self.vec_length, activation="relu")
        self.pred_dense4 = Dense(self.vec_length, activation="relu")
        self.pred_dense5 = Dense(self.vec_length, activation="relu")

        # Weights processing layers - only used if dynamic model
        if self.model_type == "dynamic":
            self.weights_dense_mean = Dense(self.timestamp - 5, activation="relu")
            self.weights_dense_std = Dense(self.timestamp - 5, activation="relu")

        # Loss function for training step
        self.mape = MeanAbsolutePercentageError()

        # If using optimizer, need to set it externally before training
        self.optimizer = None

    @staticmethod
    def FiLM(tensor, mean_weights, std_weights):
        """
        Feature-wise Linear Modulation (FiLM)
        tensor: input tensor to modulate, shape [B, T, C]
        mean_weights, std_weights: modulatory weights broadcastable to tensor
        Returns tensor * mean_weights + std_weights
        """
        return Multiply()([tensor, mean_weights]) + std_weights

    def call(self, inputs, training=False):
        """
        Forward pass.
        inputs:
          - if baseline: tensor x of shape [B, timestamp-5, vec_length]
          - if dynamic: tuple (x, weights), where
                x: [B, timestamp-5, vec_length]
                weights: [B, 2] vector for FiLM and dynamic weighting
        Returns:
          - recon: reconstruction output, shape [B, timestamp-5, vec_length]
          - predicted_future: predicted future vector, shape [B, vec_length]
          - weights (if dynamic) or None (if baseline)
        """
        if self.model_type == "dynamic":
            x, weights = inputs
            # Process weights for FiLM
            mean_weights = self.weights_dense_mean(weights)  # shape [B, timestamp-5]
            std_weights = self.weights_dense_std(weights)    # shape [B, timestamp-5]

            # Expand dims to broadcast on features:
            mean_weights = tf.expand_dims(mean_weights, axis=-1)  # [B, T, 1]
            std_weights = tf.expand_dims(std_weights, axis=-1)    # [B, T, 1]
        else:
            x = inputs
            weights = None

        # Encoder
        enc_first = self.enc_lstm1(x)
        if self.model_type == "dynamic":
            enc_first = self.FiLM(enc_first, mean_weights, std_weights)

        enc_second = self.enc_lstm2(enc_first)
        if self.model_type == "dynamic":
            enc_second = self.FiLM(enc_second, mean_weights, std_weights)

        enc_second_full = self.enc_lstm_full(enc_first)

        # Decoder for reconstruction
        dec_first = self.dec_lstm1(enc_second)
        recon = self.dec_lstm_recon(dec_first)

        # Predict future vectors through dense layers
        predicted_vec = self.pred_dense1(enc_second_full)
        predicted_vec = self.pred_dense2(predicted_vec)
        predicted_vec = self.pred_dense3(predicted_vec)
        predicted_vec = self.pred_dense4(predicted_vec)
        predicted_vec = self.pred_dense5(predicted_vec)

        if self.model_type == "dynamic":
            return recon, predicted_vec, weights
        else:
            return recon, predicted_vec

    def train_step(self, data):
        """
        Custom training loop for weighted multi-task loss.
        Data inputs:
          x: inputs as per model_type (tensor or tuple)
          y: tuple of (recon_true, future_true)
        Applies gradients with weights controlling contribution of each loss.
        """
        x, y = data
        recon_true, future_true = y
        trainable_vars = self.trainable_variables

        if self.optimizer is None:
            raise ValueError("Optimizer must be set before training!")

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # forward pass
            if self.model_type == "dynamic":
                recon_pred, future_pred, weights = y_pred
            else:
                recon_pred, future_pred = y_pred
                weights = tf.constant([[0.5, 0.5]])  # default equal weights if baseline

            # Compute separate losses
            recon_loss = self.mape(recon_true, recon_pred)
            future_loss = self.mape(future_true, future_pred)

        # Compute gradients
        recon_gradients = tape.gradient(recon_loss, trainable_vars)
        future_gradients = tape.gradient(future_loss, trainable_vars)

        # Blend gradients by weights
        # Convert weights tensor to numpy to multiply (could cause issues in graph mode;
        # this follows original user approach but can be improved)
        weights_np = weights.numpy()[0] if hasattr(weights, "numpy") else weights[0]
        combined_gradients = []
        for g_recon, g_future in zip(recon_gradients, future_gradients):
            if g_recon is not None and g_future is not None:
                combined = weights_np[0] * g_recon + weights_np[1] * g_future
            else:
                combined = g_recon if g_recon is not None else g_future
            combined_gradients.append(combined)

        # Apply gradients
        self.optimizer.apply_gradients(zip(combined_gradients, trainable_vars))

        # Optional: return losses for logging
        return {"recon_loss": recon_loss, "future_loss": future_loss}

def my_model_function():
    # Return an instance of MyModel with default parameters,
    # the caller must set optimizer before training
    model = MyModel(timestamp=300, vec_length=128, model_type="dynamic")
    return model

def GetInput():
    # Generate valid random input for MyModel:
    # For dynamic mode: tuple (x, weights)
    # - x shape: [B, timestamp-5, vec_length], e.g. B=1
    # - weights shape: [B, 2]
    B = 1
    timestamp = 300
    vec_length = 128
    x = tf.random.uniform((B, timestamp - 5, vec_length), dtype=tf.float32)
    weights = tf.random.uniform((B, 2), minval=0, maxval=1, dtype=tf.float32)
    return (x, weights)

