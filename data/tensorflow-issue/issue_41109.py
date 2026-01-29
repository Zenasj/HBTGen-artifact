# tf.random.uniform((B,), dtype=tf.string) ← Input is a batch of strings (movie-related text inputs)

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Pretrained embedding layer from TF Hub, trainable
        # Input shape: scalar string tensors (a batch of strings)
        self.hub_layer = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
            output_shape=[20],
            input_shape=[],
            dtype=tf.string,
            trainable=True,
            name="gnews_swivel_embedding"
        )
        
        # Flatten the output (though output is already 1D, but kept consistent with original)
        self.flatten = Flatten()
        
        # Dense layers with L2 regularization and Dropout
        self.dense1 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self.dropout = Dropout(0.0)  # From original, dropout 0 (disabled)
        
        # Output layer with sigmoid for multilabel classification (17 labels from y_val shape)
        self.output_layer = Dense(17, activation='sigmoid')
        
    def call(self, inputs, training=False):
        """
        Forward pass.
        Inputs: tuple/list of 4 string tensors:
          (features, plot_summary, actors_array, reviews)
        Each is a batch of strings (shape=(batch_size,))
        
        Model concatenates embeddings from all four inputs and feeds through classifier.
        """
        # Extract each input
        features_str, plot_str, actors_str, reviews_str = inputs
        
        # Embed each input with the shared pretrained embedding
        emb_features = self.hub_layer(features_str)  # shape (batch, 20)
        emb_plot = self.hub_layer(plot_str)
        emb_actors = self.hub_layer(actors_str)
        emb_reviews = self.hub_layer(reviews_str)
        
        # Concatenate all embeddings along last axis
        concat_emb = tf.concat([emb_features, emb_plot, emb_actors, emb_reviews], axis=-1)  # shape (batch, 80)
        
        x = self.flatten(concat_emb)  # flatten in case the concat_emb has extra dims
        
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        output = self.output_layer(x)  # multilabel sigmoid output
        
        return output

def my_model_function():
    """
    Create and return an instance of MyModel,
    compiled with Adam optimizer and binary crossentropy loss,
    matching the original training setup.
    """
    model = MyModel()

    # Learning rate schedule as described in the issue:
    # - Decay steps and rate inferred based on example:
    #   decay_steps=int(np.ceil((len(partial_x_train_actors_array)*0.8)//16))*1000
    # Since partial_x_train_actors_array is unknown here, use a placeholder value: 5 (from dataset size)
    decay_steps = int(tf.math.ceil((5 * 0.8) / 16)) * 1000
    # This evaluates to 0 with small numbers, but keep at least 1 to avoid zero division
    decay_steps = max(decay_steps, 1)
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.01,
        decay_steps=decay_steps,
        decay_rate=1,
        staircase=False
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def GetInput():
    """
    Generates a tuple of 4 input tensors matching the inputs expected by MyModel:
    - features: batch of movie content strings 
    - plot summary: batch of plot summary strings
    - actors_array: batch of actor lists flattened as strings (joined actor names)
    - reviews: batch of review texts strings
    
    Assumptions:
    - For simplicity, actors_array (originally array of arrays) flattened by joining actor names into single string per example.
    - Batch size set to 4 to show variable batch inputs.
    """
    
    batch_features = tf.constant([
        b'south pago pago victor mclaglen jon hall frances farmer olympe bradna gene lockhart douglass dumbrille',
        b'easy virtue jessica biel ben barnes kristin scott thomas colin firth',
        b'fragments antonin gregori derangere anouk grinberg aurelien recoing',
        b'milka film taboos milka elokuva tabuista irma huntus leena suomu'
    ], dtype=tf.string)
    
    batch_plot = tf.constant([
        b'treasure hunt adventure',
        b'young englishman marry glamorous american',
        b'psychiatrist probe traumatized soldier',
        b'small finnish lapland community milka innocent'
    ], dtype=tf.string)
    
    # The original actor arrays are arrays of bytes for each actor.
    # We'll join actor names with commas to convert array-of-arrays to one string per sample
    sample_actors = [
        b'victor mclaglen, jon hall, frances farmer, olympe bradna',
        b'jessica biel, ben barnes, kristin scott thomas, colin firth',
        b'gregori derangere, anouk grinberg, aurelien recoing, niels arestrup',
        b'irma huntus, leena suomu, matti turunen, eikka lehtonen'
    ]
    batch_actors = tf.constant(sample_actors, dtype=tf.string)
    
    batch_reviews = tf.constant([
        b'edward small take director alfred e green brilliant assemblage',
        b'jessica biel probably best know virtuous good girl preacher kid',
        b'saw night eurocine event movie european country show day',
        b'rauni mollberg earth sinful song favorite foreign film establish director'
    ], dtype=tf.string)
    
    return (batch_features, batch_plot, batch_actors, batch_reviews)

