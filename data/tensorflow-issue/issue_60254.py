# tf.random.uniform((B,), dtype=tf.string) ← The model input is a batch of variable-length strings (text samples)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Vocabulary size limit
        self.VOCAB_SIZE = 20000

        # Text vectorization layer: standardizes text to lowercase, tokenizes, and creates integer sequences
        # Note: adapted later on some dataset input in __init__ for minimal reproduction here
        self.encoder = tf.keras.layers.TextVectorization(
            standardize='lower',
            max_tokens=self.VOCAB_SIZE,
            output_mode='int',  # output integer indices
            output_sequence_length=None  # variable length, uses masking downstream
        )
        # Set up embedding layer with mask_zero=True to handle padding/masked tokens
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.VOCAB_SIZE + 2,  # +2 to safely include vocab size + oov + padding tokens
            output_dim=64,
            mask_zero=True
        )
        # Bi-directional LSTM layer with 128 units each direction
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        
    def adapt_encoder(self, text_dataset):
        # Adapt the text vectorization layer on the training text
        self.encoder.adapt(text_dataset)
        # Update embedding input_dim to match encoder vocabulary size (+2)
        vocab_len = len(self.encoder.get_vocabulary())
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_len,
            output_dim=64,
            mask_zero=True
        )
    
    def call(self, inputs, training=False):
        # inputs: batch of strings shape=(batch_size,)
        x = self.encoder(inputs)  # shape (batch_size, sequence_length), int indices
        x = self.embedding(x)     # (batch_size, seq_len, 64)
        x = self.bi_lstm(x)       # (batch_size, 256)
        x = self.dense1(x)        # (batch_size, 64)
        output = self.dense2(x)   # (batch_size, 1), logits for binary classification
        return output

def my_model_function():
    # Instantiate model and adapt encoder on minimal example training text dataset
    model = MyModel()
    # For the sake of a self-contained example, we create a minimal dataset
    # Assumption: The model expects input strings similar to the example dataset in the issue.
    example_texts = tf.data.Dataset.from_tensor_slices([
        'Россия',
        'Вчера смотрел в кино - потрясающий фильм! Актёры высшие, невероятные декорации, безудержный драйв на протяжении всего фильма. Давно не испытывал такого восторга от просмотра! 10/10',
        'Норм фильм,в своём стиле не понимаю что другие ожидали))одно смутило когда сцена в клубе все танчили пока бойня была типо ниче не замечая а как картежника завалили все с истериками побежали,типа хуясе тут все в настаящую))))да и пёсель зачетный))',
        'Да пипец блин, меня хватило на 10 минут. Это днище',
        'Бредовый фильм не советую'
    ])
    model.adapt_encoder(example_texts.batch(2))
    # Compile model for binary classification
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random batch of strings similar in style to the training data
    # Batch size chosen arbitrarily as 4 for demonstration; can be tuned.
    sample_texts = [
        'меня хватило на 10 минут',
        'Классный фильм',
        'Фильм говно',
        'Не советую к просмотру'
    ]
    # Convert to tf.Tensor dtype string, shape (4,)
    return tf.constant(sample_texts)

