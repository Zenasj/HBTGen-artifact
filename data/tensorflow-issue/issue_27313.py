# tf.random.uniform((B, 650), dtype=tf.float32) ← Input shape inferred from X_train and X_test in issue (batch varies, feature dim 650)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, sequence_len=650, classes=10000):
        """
        Reconstructed model based on the Conv1D architecture described in the issue.
        Input shape: (batch_size, sequence_len=650)
        Output shape: (batch_size, 1) regression output (linear activation)
        
        Note: `classes` is set to a default 10000 but unused in current embedding, kept for similarity 
        with original code snippet naming but embedding input dim clarified later as well.
        
        Here, the input is a sequence of token indices, with sequence_len=650.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=classes, output_dim=8, input_length=sequence_len)
        
        # Conv1D stack as per description with BatchNorm + ReLU activations and pooling
        self.conv1 = tf.keras.layers.Conv1D(128, 7, padding='valid')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        
        self.conv2 = tf.keras.layers.Conv1D(128, 3, padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        
        self.conv3 = tf.keras.layers.Conv1D(128, 3, padding='valid')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')
        
        self.pool = tf.keras.layers.MaxPooling1D(3)
        
        self.conv4 = tf.keras.layers.Conv1D(256, 3, padding='valid')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.act4 = tf.keras.layers.Activation('relu')
        
        self.conv5 = tf.keras.layers.Conv1D(256, 3, padding='valid')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.act5 = tf.keras.layers.Activation('relu')
        
        self.conv6 = tf.keras.layers.Conv1D(256, 3, padding='valid')
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.act6 = tf.keras.layers.Activation('relu')
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        
        self.dense1 = tf.keras.layers.Dense(256)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.act7 = tf.keras.layers.Activation('relu')
        
        self.dense2 = tf.keras.layers.Dense(1)
        self.final_act = tf.keras.layers.Activation('linear')
    
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        inputs: tf.int32 tensor of shape (batch_size, 650), token indices
        returns: tf.float32 tensor of shape (batch_size, 1), regression output
        """
        x = self.embedding(inputs)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.act4(x)
        
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.act5(x)
        
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.act6(x)
        
        x = self.global_pool(x)
        
        x = self.dense1(x)
        x = self.bn7(x, training=training)
        x = self.act7(x)
        
        x = self.dense2(x)
        x = self.final_act(x)
        
        return x

def my_model_function():
    """
    Return an instance of MyModel.
    The sequence_len=650 and classes=some large integer are assumed from original code snippet.
    """
    SEQUENCE_LEN = 650
    CLASSES = 10000  # placeholder, adjust if needed
    return MyModel(sequence_len=SEQUENCE_LEN, classes=CLASSES)

def GetInput():
    """
    Return a random input tensor compatible with MyModel.
    The model expects integer token indices in shape (batch_size, 650). 
    The tokens should be within [0, CLASSES).
    For demonstration, generate token indices uniformly randomly.
    """
    batch_size = 128  # Typical batch size divisible by TPU cores (e.g., 8, 128, 1024)
    sequence_len = 650
    classes = 10000  # Matches the embedding input_dim
    
    # Random integers for token indices in range [0, classes)
    inp = tf.random.uniform(shape=(batch_size, sequence_len), minval=0, maxval=classes, dtype=tf.int32)
    return inp

