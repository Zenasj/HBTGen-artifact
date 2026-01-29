# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from Input((10,)) in shared model

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import tensorflow.keras.backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Shared base layers
        self.shared_dense1 = Dense(64, activation='relu')
        self.shared_dense2 = Dense(128, activation='relu')

        # Actor model layers
        self.actor_dense = Dense(128, activation='relu')
        self.actor_output = Dense(5, activation='softmax')

        # Critic model layers
        self.critic_dense = Dense(128, activation='relu')
        self.critic_output = Dense(1, activation='linear')

        # Optimizer
        self.rms_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, epsilon=0.1, rho=0.99)

    def call(self, inputs):
        """
        Forward pass returns actor and critic outputs.
        """
        x = self.shared_dense1(inputs)
        x = self.shared_dense2(x)

        actor_x = self.actor_dense(x)
        actor_out = self.actor_output(actor_x)

        critic_x = self.critic_dense(x)
        critic_out = self.critic_output(critic_x)

        return actor_out, critic_out

    def compute_actor_loss(self, action_pl, advantages_pl, actor_out):
        """
        Computes the A3C actor loss.
        """
        # Sum over actions weighted by action_pl (one-hot or soft labels)
        weighted_actions = tf.reduce_sum(action_pl * actor_out, axis=1)
        # Eligibility term using log probability weighted by advantages; stop gradients on advantages
        eligibility = tf.math.log(weighted_actions + 1e-10) * tf.stop_gradient(advantages_pl)
        # Entropy term encourages exploration
        entropy = tf.reduce_sum(actor_out * tf.math.log(actor_out + 1e-10), axis=1)
        # Final loss: entropy scaled plus negative eligibility sum, averaged over batch
        loss = tf.reduce_mean(0.001 * entropy - eligibility)
        return loss

    def get_actor_updates(self, action_pl, advantages_pl):
        """
        Generates the RMSprop updates for the actor weights using the proper updated get_updates signature.
        In TensorFlow 2.x Keras, it's recommended to use tf.GradientTape instead;
        This is a reconstruction mimicking original get_updates behavior for compatibility.
        """
        with tf.GradientTape() as tape:
            actor_out, _ = self.call(self._last_input)
            loss = self.compute_actor_loss(action_pl, advantages_pl, actor_out)
        # Compute gradients and create train op
        grads = tape.gradient(loss, self.trainable_variables)
        # Apply gradients via optimizer
        updates = self.rms_optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return updates

def my_model_function():
    """
    Return an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a valid input tensor that matches MyModel input expectations (batch size arbitrary).
    Shape: (batch_size, 10)
    """
    batch_size = 4  # Example batch size
    return tf.random.uniform((batch_size, 10), dtype=tf.float32)

