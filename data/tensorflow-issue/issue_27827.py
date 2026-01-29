# tf.random.uniform((B, 84, 84, 4), dtype=tf.float32) ← Input shape inferred from keras/eager models (batch, height, width, channels)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

        # Conv layers with BatchNorm and ReLU
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, padding='valid', activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-5)
        self.act1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='valid', activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-5)
        self.act2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='valid', activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-5)
        self.act3 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()

        # Dense layers with ReLU and Dropout as per original keras model
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.05)

        # Final logits layer (no activation)
        self.logits_layer = tf.keras.layers.Dense(self.action_space, activation=None)

    def call(self, inputs, training=False):
        # Expecting inputs as tuple/list: (images, rewards)
        x, r = inputs  # images: [B, 84, 84, 4], r: [B, 1] or scalar reward tensor

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        logits = self.logits_layer(x)  # [B, action_space]
        return logits

    @staticmethod
    def policy_loss(r):
        # r: tensor with shape [B, 1] or [B], advantage or reward signal

        def loss(labels, logits):
            # labels: one-hot or probability distribution for actions, logits: model output logits
            policy = tf.nn.softmax(logits)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)  # note: cross_entropy for entropy calculation
            log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # p_loss = log * stop_gradient(r) - 0.01*entropy
            p_loss = log_prob * tf.stop_gradient(r)
            p_loss = p_loss - 0.01 * entropy
            total_loss = tf.reduce_mean(p_loss)
            return total_loss

        return loss


def my_model_function():
    # For demonstration, let's assume action_space=3 (e.g., 3 discrete actions)
    # This parameter can be adjusted to fit the specific use case
    action_space = 3
    model = MyModel(action_space=action_space)

    # Compile the model here with Adam optimizer and the custom loss wrapped with policy_loss
    # We create a dummy rewards tensor as input shape (batch_size,1) is required for loss closure
    rewards_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='rewards')

    # Build model inputs placeholders for compilation (images and rewards)
    images_input = tf.keras.Input(shape=(84, 84, 4), dtype=tf.float32, name='images')

    # Get logits output
    logits = model([images_input, rewards_input])

    # Create a tf.keras.Model to enable compile(). This model is used only for compilation/training.
    keras_model = tf.keras.Model(inputs=[images_input, rewards_input], outputs=logits)

    # Use the loss closure function to create a compatible loss function
    # We pass rewards_input tensor to get the loss function that depends on r

    loss_fn = MyModel.policy_loss(rewards_input)

    # Use Adam optimizer with a default LR (can be adjusted later)
    adam_optimizer = tf.keras.optimizers.Adam()

    keras_model.compile(optimizer=adam_optimizer, loss=loss_fn)

    # Set the compiled keras_model inside our MyModel instance for training convenience
    model.keras_compiled_model = keras_model

    return model


def GetInput():
    # Return a tuple of two tensors:
    # - images: random tensor with shape (batch_size, 84, 84, 4), dtype float32
    # - rewards: random tensor (batch_size, 1), float32, required for loss function

    batch_size = 2  # small batch for demonstration; can be changed as needed
    images = tf.random.uniform((batch_size, 84, 84, 4), dtype=tf.float32)
    rewards = tf.ones((batch_size, 1), dtype=tf.float32)  # e.g. rewards of 1 for all batch items

    return (images, rewards)

