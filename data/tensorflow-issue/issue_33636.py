# tf.random.uniform((B, D), dtype=tf.float32) where B=batch size, D=number of labels/classes

import tensorflow as tf

class MultiLabelMacroSpecificity(tf.keras.metrics.Metric):
    """
    Multi-label Macro Specificity Metric.
    Computes specificity metric adapted for multi-label classification tasks.
    Operates on raw prediction probabilities (y_pred) and ground truth labels (y_true).
    
    Arguments:
      threshold: float, decision threshold for positive prediction (default 0.5).
      name: string, metric name.
      dtype: data type for metric variables.
    """

    def __init__(self, name='multi_label_macro_specificity', threshold=0.5, **kwargs):
        super(MultiLabelMacroSpecificity, self).__init__(name=name, **kwargs)
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        
        # Accumulator variables as float32 to keep sum over batches
        self.true_negatives = self.add_weight(name='tn', initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.total_updates = self.add_weight(name='updates', initializer='zeros', dtype=tf.float32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true and y_pred expected shape: (batch_size, num_labels)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Threshold predictions and labels to get boolean positive/negative
        pred_is_pos = tf.greater(y_pred, self.threshold)
        pred_is_neg = tf.logical_not(pred_is_pos)

        label_is_pos = tf.greater(y_true, self.threshold)
        label_is_neg = tf.logical_not(label_is_pos)

        # Calculate counts for true negatives and false positives across all labels and batch
        tn = tf.reduce_sum(tf.cast(tf.logical_and(pred_is_neg, label_is_neg), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(pred_is_pos, label_is_neg), tf.float32))

        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.total_updates.assign_add(1.0)
        
        # No weighting or averaging here; result() will compute final metric.
        
    def result(self):
        # Compute specificity = TN / (TN + FP), safe divide
        return tf.math.divide_no_nan(self.true_negatives, self.true_negatives + self.false_positives)
        
    def reset_states(self):
        # Reset all accumulators to zero
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.total_updates.assign(0.0)

class MyModel(tf.keras.Model):
    """
    A model exposing both raw logits and sigmoid outputs,
    supporting multi-label scenarios.

    Forward pass outputs a dict:
      'logits': sigmoid probabilities for metrics,
      'raw_out_for_loss': raw logits for computing loss with sigmoid_cross_entropy.
    """

    def __init__(self, num_labels=10):
        super(MyModel, self).__init__()
        self.num_labels = num_labels

        # Simple example network architecture:
        # For demonstration, just a single Dense layer producing logits.
        self.dense = tf.keras.layers.Dense(self.num_labels, activation=None)
        
    def call(self, inputs, training=False):
        logits = self.dense(inputs)  # raw logits, shape (batch_size, num_labels)
        probs = tf.sigmoid(logits)   # probabilities for multi-label
        return {
            'logits': probs,                # to be used for metrics
            'raw_out_for_loss': logits      # to be used for loss computation
        }

def my_model_function():
    # Create and return MyModel instance with some default number of labels =10
    # This can be adjusted or passed as argument if needed
    return MyModel(num_labels=10)

def GetInput():
    # Generate random float input tensor matching expected input shape.
    # Assume input dimension = 20 (feature size), batch size = 32
    # Adjust dims if needed based on actual model input requirements.
    batch_size = 32
    feature_dim = 20
    # Use uniform distribution in [0,1)
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

