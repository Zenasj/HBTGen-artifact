# tf.random.uniform((batch_size, ), dtype=tf.int32), tf.random.uniform((batch_size, 21), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_users=1000, num_item_features=21, embedding_dim=32, num_outputs=32):
        """
        Args:
            num_users: Total distinct users (vocabulary for embedding)
            num_item_features: Number of features describing each movie/item
            embedding_dim: Dimensionality of user embedding vector (K)
            num_outputs: Output size from user/item subnetworks
        """
        super().__init__()

        # User embedding layer
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)

        # Subnetwork for user representation
        self.user_NN = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation='linear'),
        ])

        # Subnetwork for item representation
        self.item_NN = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation='linear'),
        ])

        # Dot product layer as output
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs, training=False):
        """
        Args:
            inputs: tuple (user_ids, item_features)
                user_ids: tf.Tensor of shape (batch_size,) with dtype int32/int64
                item_features: tf.Tensor of shape (batch_size, num_item_features), float32
            training: boolean, to toggle dropout behavior
        Returns:
            similarity scores (dot product) between user and item embeddings.
        """
        user_ids, item_features = inputs

        # Embed users - result shape: (batch_size, embedding_dim)
        user_emb = self.user_embedding(user_ids)  # (B, embedding_dim)
        
        # Flatten possible extra dims (user input shape (1,) embedded to (B, 1, embedding_dim))
        if len(user_emb.shape) == 3:
            user_emb = tf.squeeze(user_emb, axis=1)

        # Pass embeddings through user NN
        vu = self.user_NN(user_emb, training=training)
        vu = tf.linalg.l2_normalize(vu, axis=1)

        # Pass item features through item NN
        vm = self.item_NN(item_features, training=training)
        vm = tf.linalg.l2_normalize(vm, axis=1)

        # Dot product similarity
        scores = self.dot([vu, vm])  # shape (batch_size, 1)
        # Flatten output shape to (batch_size,) for loss compatibility
        return tf.squeeze(scores, axis=1)


def my_model_function(num_users=1000, num_item_features=21, embedding_dim=32, num_outputs=32):
    # Return an instance of MyModel with given or default params
    model = MyModel(num_users=num_users,
                    num_item_features=num_item_features,
                    embedding_dim=embedding_dim,
                    num_outputs=num_outputs)

    # Compile with mean squared error loss as it's a regression task
    model.compile(optimizer='adam', loss='mse')
    return model


def GetInput(batch_size=256, num_users=1000, num_item_features=21):
    """
    Generate random example inputs compatible with MyModel:

    - user_ids: random integers [0, num_users)
    - item_features: random floats for movie/item features
    """
    user_ids = tf.random.uniform(shape=(batch_size,), minval=0, maxval=num_users, dtype=tf.int32)
    item_features = tf.random.uniform(shape=(batch_size, num_item_features), dtype=tf.float32)
    return (user_ids, item_features)

