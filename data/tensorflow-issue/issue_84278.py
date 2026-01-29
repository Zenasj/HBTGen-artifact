# tf.random.uniform((1280, 10000), dtype=tf.int32) for title and text_body inputs,
# and tf.random.uniform((1280, 100), dtype=tf.int32) for tags input (binary vectors)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vocabulary_size = 10000
        self.num_tags = 100
        self.num_departments = 4
        
        # Define layers matching the described architecture
        self.concat = layers.Concatenate()
        self.dense64 = layers.Dense(64, activation="relu")
        self.priority_head = layers.Dense(1, activation="sigmoid", name="priority")
        self.department_head = layers.Dense(self.num_departments, activation="softmax", name="department")

    def call(self, inputs, training=False):
        # Inputs is expected to be a dict with keys: 'title', 'text_body', 'tags'
        # The inputs are all binary (0 or 1) vectors of specified shapes.
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        # Concatenate inputs along last dimension
        features = self.concat([title, text_body, tags])
        features = self.dense64(features)
        priority = self.priority_head(features)
        department = self.department_head(features)
        
        return {"priority": priority, "department": department}

def my_model_function():
    model = MyModel()
    # Compile the model to match the setting in the issue
    model.compile(
        optimizer="rmsprop",
        loss={
            "priority": "mean_squared_error",
            "department": "categorical_crossentropy"
        },
        metrics={
            "priority": ["mean_absolute_error"],
            "department": ["accuracy"]
        }
    )
    return model

def GetInput():
    # Generate inputs matching the required input shapes and dtypes as binary vectors (0 or 1)
    num_samples = 32  # Using smaller batch size for example and compilation convenience
    vocabulary_size = 10000
    num_tags = 100

    # Use tf.random.uniform with dtype=tf.int32, then cast to float32 to simulate binary multi-hot vectors
    title = tf.cast(tf.random.uniform((num_samples, vocabulary_size), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    text_body = tf.cast(tf.random.uniform((num_samples, vocabulary_size), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    tags = tf.cast(tf.random.uniform((num_samples, num_tags), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    
    return {"title": title, "text_body": text_body, "tags": tags}

