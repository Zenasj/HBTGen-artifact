# tf.random.uniform((B, total_words), dtype=tf.float32) ← Input shape inferred as (batch_size=1, total_words), batch size 1 since processing one sentence at a time

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, dimVectors, total_words):
        super().__init__()
        # Model as described: Sequential with one hidden Dense(20, relu) and output Dense(5, softmax)
        self.dense1 = tf.keras.layers.Dense(20, activation='relu', input_shape=(dimVectors,))
        self.dense2 = tf.keras.layers.Dense(5, activation='softmax')
        
        # The wordVectors embedding matrix must be a variable of shape (total_words, dimVectors)
        # Initialized here as trainable variable with random normal initializer as placeholder
        self.wordVectors = tf.Variable(
            tf.random.normal(shape=(total_words, dimVectors), dtype=tf.float32),
            trainable=True,
            name='wordVectors'
        )
    
    @tf.function(jit_compile=True)
    def call(self, one_hot_sentences):
        """
        one_hot_sentences: tf.Tensor of shape (sentence_length, total_words), dtype float32
        representing one-hot encoding of words in a sentence
        
        Processing steps:
        - Multiply one_hot_sentences by wordVectors to get feature vectors for each word
        - Sum those feature vectors along sentence length dim → (1, dimVectors)
        - Pass through Dense layers to get output (1, 5)
        """
        # Compute embedding features for each word (sentence_length, dimVectors)
        feature = tf.matmul(one_hot_sentences, self.wordVectors)
        # Sum over sentence words → shape (1, dimVectors)
        feature_sum = tf.reduce_sum(feature, axis=0, keepdims=True)
        # Pass through model layers
        x = self.dense1(feature_sum)
        y_pred = self.dense2(x)
        return y_pred
    
    def train_step(self, one_hot_sentences, label, optimizer):
        """
        Single optimization step on one sentence
        
        Args:
        - one_hot_sentences: tf.Tensor (sentence_length, total_words)
        - label: tf.Tensor (5, 1), one-hot label for the sentence, reshaped as column vector
        - optimizer: tf.keras.optimizers.Optimizer instance
        
        Returns:
        - loss scalar tensor
        """
        with tf.GradientTape() as tape:
            y_pred = self.call(one_hot_sentences)
            # MeanSquaredError used per original example
            loss = tf.keras.losses.MeanSquaredError()(label, y_pred)
        
        # Gradients with respect to wordVectors and model trainable variables
        gradients = tape.gradient(loss, [self.wordVectors] + self.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, [self.wordVectors] + self.trainable_variables))
        return loss

def my_model_function():
    """
    Returns an instance of MyModel
    
    Assumptions:
    - dimVectors: embedding vector dimension (inferred from original example as a placeholder, e.g. 100)
    - total_words: size of vocabulary (passed or assumed, e.g. 10000)
    These need to be replaced or passed appropriately in a real scenario.
    """
    # Placeholder values - these should be replaced to match the intended dataset
    dimVectors = 100      # embedding dimension
    total_words = 10000   # vocabulary size
    model = MyModel(dimVectors, total_words)
    return model

def GetInput():
    """
    Returns a random one-hot encoded sentence tensor compatible with MyModel.
    Since MyModel call expects shape (sentence_length, total_words), generate a batch of one sentence:
    
    - sentence_length: random length between 5 and 15 (typical sentence length)
    - total_words: must match model's total_words (here 10000)
    
    Generate random indices and create one-hot tensor accordingly.
    """
    import numpy as np
    
    sentence_length = np.random.randint(5, 16)
    total_words = 10000  # must match the initialization in my_model_function
    
    indices = np.random.choice(total_words, size=sentence_length, replace=True)
    # Create one-hot matrix shape (sentence_length, total_words)
    one_hot_np = np.zeros((sentence_length, total_words), dtype=np.float32)
    one_hot_np[np.arange(sentence_length), indices] = 1.0
    
    return tf.convert_to_tensor(one_hot_np, dtype=tf.float32)

