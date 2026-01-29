# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape based on the example: batch size B, input features 2

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple functional like model inside subclassing
        # Example from issue: Input shape (2,), output Dense(2) with softmax activation
        
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
        
        # For demonstration, imagine we want to support both "vanilla" Model output
        # and a "wrapped" behavior that would otherwise rely on predict_classes in older wrappers.
        # Here, emulate predict_classes logic as outputting the predicted class indices.
    
    def call(self, inputs, training=False):
        # Forward pass producing probabilities
        probs = self.dense(inputs)
        return probs

    def predict_classes(self, inputs):
        # Emulate the legacy predict_classes (argmax over softmax output)
        probs = self.call(inputs, training=False)
        return tf.argmax(probs, axis=-1)
    
    # We can add a method that mimics the sklearn wrapper expected predict behavior:
    def predict(self, inputs):
        # Return class indices (integers) similar to predict_classes
        return self.predict_classes(inputs)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Since functional models were compiled in sample code, compile this as well
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor input matching (batch_size, 2)
    # Let's assume a batch size of 4 for demonstration
    batch_size = 4
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)

