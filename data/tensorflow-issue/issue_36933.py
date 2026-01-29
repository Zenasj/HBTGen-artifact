# tf.random.uniform((B,)) for 'movie' input and tf.random.uniform((B,)) for 'user' input
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two independent dense layers to mimic embeddings for movie and user inputs
        self.movie_dense = tf.keras.layers.Dense(units=40, activation='relu')
        self.user_dense = tf.keras.layers.Dense(units=40, activation='relu')
        # Final dense layer to produce sigmoid output from concatenated embeddings
        self.output_dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
    
    def call(self, inputs):
        # Expecting inputs as a dict with keys 'movie' and 'user'
        movie_input = inputs['movie']
        user_input = inputs['user']
        # Apply the dense layers to their respective inputs
        movie_embed = self.movie_dense(movie_input)
        user_embed = self.user_dense(user_input)
        # Concatenate embeddings along feature axis (axis=1)
        combined = tf.concat([movie_embed, user_embed], axis=1)
        # Final output layer
        output = self.output_dense(combined)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a batch of inputs similar to the toy_data example:
    # movie: shape (batch_size, 1), user: shape (batch_size, 1)
    batch_size = 2
    # Random integers for movie IDs between 0 and 9 (inclusive)
    movie_input = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=10, dtype=tf.int32)
    # Convert to float32 since Dense layers expect floats
    movie_input = tf.cast(movie_input, tf.float32)
    # Random integers for user IDs between 0 and 20 (inclusive)
    user_input = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=20, dtype=tf.int32)
    user_input = tf.cast(user_input, tf.float32)
    return {'movie': movie_input, 'user': user_input}

