# tf.random.uniform((BATCH_SIZE, 77), dtype=tf.int32) ← Assumed input shape for "tokens" input to text encoder

import tensorflow as tf

# Placeholder for the actual TextEncoder model from keras_cv.models.stable_diffusion.text_encoder
# Since the original TextEncoder code isn't provided in the issue, we create a minimal
# stand-in model that mimics expected input/output shapes and behavior for demonstration.
class TextEncoder(tf.keras.Model):
    def __init__(self, max_prompt_length=77):
        super().__init__()
        self.max_prompt_length = max_prompt_length
        
        # For simplicity, using embeddings + BiLSTM to simulate encoding behavior:
        self.token_embedding = tf.keras.layers.Embedding(input_dim=49408, output_dim=768)  # vocab size approx
        self.position_embedding = tf.keras.layers.Embedding(input_dim=max_prompt_length, output_dim=768)
        self.encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True))
        # Final projection layer
        self.projection = tf.keras.layers.Dense(768)

    def call(self, inputs, training=False):
        # Expect inputs as list or tuple: [tokens, pos_ids]
        tokens, pos_ids = inputs
        
        # tokens: (B, max_prompt_length), pos_ids: (1, max_prompt_length)
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(tokens)  # (B, max_prompt_length, 768)
        pos_embeds = self.position_embedding(pos_ids)  # (1, max_prompt_length, 768)
        
        # Broadcast pos_embeds to batch size
        pos_embeds = tf.broadcast_to(pos_embeds, tf.shape(token_embeds))
        
        x = token_embeds + pos_embeds
        x = self.encoder_lstm(x, training=training)  # (B, max_prompt_length, 1536)
        
        output = self.projection(x)  # (B, max_prompt_length, 768)
        return output


class MyModel(tf.keras.Model):
    def __init__(self, max_prompt_length=77, batch_size=3):
        super().__init__()
        self.max_prompt_length = max_prompt_length
        self.batch_size = batch_size
        # Instantiate the underlying text encoder model
        self.text_encoder = TextEncoder(max_prompt_length)
        
        # Constants from stable diffusion repo for unconditional tokens, represented as int32 tensor
        # Since original _UNCONDITIONAL_TOKENS is not provided here, we assume a placeholder token sequence of length 77.
        self.UNCONDITIONAL_TOKENS = tf.constant([[49407] * max_prompt_length], dtype=tf.int32)  # EOS or pad token
        
        # Precompute position ids tensor with shape (1, max_prompt_length)
        self.POS_IDS = tf.constant([list(range(max_prompt_length))], dtype=tf.int32)
        
    def call(self, inputs, training=False):
        """
        inputs: a dict with key "tokens" containing a int32 tensor of shape (batch_size, max_prompt_length)
        Outputs a dict with keys:
         - 'context': encoded representations for the input tokens, shape (batch_size, max_prompt_length, feature_dim)
         - 'unconditional_context': encoded representations for the unconditional tokens, shape (batch_size, max_prompt_length, feature_dim)
        """
        tokens = inputs["tokens"]  # shape (B, max_prompt_length)
        
        # Encode the input prompt tokens with positions
        encoded_text = self.text_encoder([tokens, self.POS_IDS], training=training)  # (B, max_prompt_length, 768)
        
        # Squeeze if rank == 2 (ambiguous in original code), but here our output always rank 3, so skip
        
        # Encode the unconditional tokens - shape (1, max_prompt_length)
        unconditional_context = self.text_encoder([self.UNCONDITIONAL_TOKENS, self.POS_IDS], training=training)  # (1, max_prompt_length, 768)

        # Repeat unconditional context to batch size along axis 0
        unconditional_context = tf.repeat(unconditional_context, tf.shape(tokens)[0], axis=0)  # (B, max_prompt_length, 768)
        
        return {
            "context": encoded_text,
            "unconditional_context": unconditional_context,
        }


def my_model_function():
    # Return an instance of MyModel with default parameters matching original code setup
    return MyModel()


def GetInput():
    # Return a random int32 tensor simulating "tokens" input with shape (batch_size, max_prompt_length)
    batch_size = 3
    max_prompt_length = 77
    # Simulate input with random token ids below vocab size (49408), typical for clip tokenizers
    tokens = tf.random.uniform(
        shape=(batch_size, max_prompt_length),
        minval=0,
        maxval=49407,
        dtype=tf.int32,
    )
    return {"tokens": tokens}

