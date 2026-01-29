# tf.random.uniform((16, 5), dtype=tf.float32) ← inferred input shape and dtype from example code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the actor and critic subnetworks
        # Assumptions:
        # - Input shape for state is (5,)
        # - Actor outputs an action vector with shape (1,) (same as Dense(1))
        # - Critic takes as input the concatenation of (action, state) and outputs scalar value
        
        # Actor network
        self.actor_dense = tf.keras.layers.Dense(1, activation='relu', name="actor_dense")
        
        # Critic network: two inputs (action, state)
        # We'll implement critic as a single sequential module for simplicity:
        # As outlined in comments, critic input is concatenation of actor output and state.
        self.critic_dense1 = tf.keras.layers.Dense(16, activation='relu', name="critic_dense1")
        self.critic_dense2 = tf.keras.layers.Dense(1, name="critic_dense2")  # output single value
        
        # Optimizers for actor and critic
        self.actor_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()
        
    def call(self, inputs):
        """
        Forward pass through actor and critic.
        inputs: a tuple (state), or just a single tensor for state
        
        Returns:
            actor_output: action prediction from actor
            critic_output: value prediction from critic given (action, state)
        """
        state = inputs
        # Actor forward pass
        action = self.actor_dense(state)
        
        # Critic forward pass: concatenate action and state along last axis
        critic_input = tf.concat([action, state], axis=-1)
        x = self.critic_dense1(critic_input)
        value = self.critic_dense2(x)
        
        return action, value

    @tf.function(jit_compile=True)
    def train_critic(self, state, action, target):
        """
        One training step for critic network, minimizing MSE between critic(state, action) and target Q values.
        """
        with tf.GradientTape() as tape:
            critic_input = tf.concat([action, state], axis=-1)
            q_pred = self.critic_dense2(self.critic_dense1(critic_input))
            loss = tf.reduce_mean(tf.square(target - q_pred))
        grads = tape.gradient(loss, self.critic_dense1.trainable_variables + self.critic_dense2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic_dense1.trainable_variables + self.critic_dense2.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def train_actor(self, state):
        """
        One training step for actor network using policy gradient:
        Maximize the critic's output by adjusting actor's weights.
        We minimize -mean(critic_output).
        """
        with tf.GradientTape() as tape:
            action = self.actor_dense(state)
            critic_input = tf.concat([action, state], axis=-1)
            q_val = self.critic_dense2(self.critic_dense1(critic_input))
            actor_loss = -tf.reduce_mean(q_val)
        grads = tape.gradient(actor_loss, self.actor_dense.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor_dense.trainable_variables))
        return actor_loss

def my_model_function():
    # Return an instance of MyModel with optimizer initialized
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input is (batch_size, 5), batch size inferred as 16 from examples
    return tf.random.uniform((16, 5), dtype=tf.float32)

