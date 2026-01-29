# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← input shape from CIFAR-10 dataset used in model

import tensorflow as tf
from distutils.version import LooseVersion

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Load or define the backbone classifier model (ResNet50) 
        # with input shape (32,32,3), no pre-trained weights, max pooling, and 10 classes
        # taking into account API differences across TF versions.
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            base_model = tf.keras.applications.ResNet50(include_top=True, 
                                                       weights=None, 
                                                       input_shape=(32,32,3), 
                                                       pooling='max', 
                                                       classes=10)
        else:
            base_model = tf.keras.applications.resnet.ResNet50(include_top=True, 
                                                               weights=None, 
                                                               input_shape=(32,32,3), 
                                                               pooling='max', 
                                                               classes=10)

        # Define inputs
        self.clf_input = tf.keras.layers.Input(shape=(32,32,3), name="model_input")
        self.label_ref = tf.keras.layers.Input(shape=(10,), name="label_ref")

        # Forward pass through base model
        self.clf_out = base_model(self.clf_input)

        # Save model components for use in call
        self.base_model = base_model

        # Following the original "cw_loss" logic, we store necessary tensors
        # Note: this model instance is designed for inference/training usage consistent with 
        # the loss function defined below.

    def call(self, inputs, training=False):
        """
        Forward call expects inputs as a tuple (images, labels), where:
          - images: Tensor, shape (B, 32, 32, 3)
          - labels: Tensor, shape (B, 10) one-hot encoded
        
        Returns logits from the ResNet50 base_model.
        """
        # Unpack inputs tuple (image batch and labels batch)
        x, label_ref = inputs
        
        # Forward through base classification model
        logits = self.base_model(x, training=training)
        return logits

    @staticmethod
    def cw_loss(label_ref, clf_out):
        """
        Implementation of the custom Carlini-Wagner like loss function (cw_loss) as discussed.
        
        Essentially:
          - Computes difference of logits to true class logit plus margin;
          - Applies ReLU, softmax weighting to focus on classes differing from true class;
          - Computes weighted sum as loss per example;
          - Returns mean loss over batch.

        Args:
            label_ref: ground-truth one-hot labels tensor, shape (B,10)
            clf_out: logits tensor from model, shape (B,10)

        Returns:
            scalar tensor loss
        """
        # Handle 'keepdims' argument name difference for TF < 1.14 compatibility
        keepdims_arg = 'keepdims' if LooseVersion(tf.__version__) >= LooseVersion('1.14.0') else 'keep_dims'

        # Compute correct class logits: sum over label * logits reduces to correct class logits
        correct_logit = tf.reduce_sum(label_ref * clf_out, axis=1, **{keepdims_arg: True})

        # Calculate distance with margin: difference between other logits & correct logit plus margin (10 for non-true classes)
        distance = tf.nn.relu(clf_out - correct_logit + (1 - label_ref) * 10)

        # Identify inactive positions (distance very small)
        inactivate = tf.cast(tf.less_equal(distance, 1e-9), tf.float32)

        # Softmax weighting: large negative on inactive positions to zero out in softmax
        weight_input = -1e9 * inactivate + distance
        weight = tf.keras.layers.Activation('softmax')(weight_input)

        # Weighted sum over non-true classes of distance * weight
        loss_vec = tf.reduce_sum((1 - label_ref) * distance * weight, axis=1)

        # Mean loss across batch
        loss = tf.reduce_mean(loss_vec)
        return loss

def my_model_function():
    """
    Construct and return an instance of MyModel.
    For practical training/inference as in discussed example,
    the model expects input as a tuple (images, labels).
    """
    return MyModel()

def GetInput():
    """
    Generate a random input compatible with MyModel's expected input:
      - images tensor of shape (batch_size, 32, 32, 3), float32
      - labels tensor of shape (batch_size, 10), one-hot encoded

    Here, we use batch_size=4 arbitrarily for demonstration.
    """
    import numpy as np
    
    batch_size = 4
    # Random float image batch in [0,1]
    images = tf.random.uniform(shape=(batch_size, 32, 32, 3), dtype=tf.float32)
    
    # Random integer labels in [0,9]
    labels_int = np.random.randint(low=0, high=10, size=(batch_size,))
    # One-hot encode
    labels = tf.one_hot(labels_int, depth=10, dtype=tf.float32)
    
    return (images, labels)

