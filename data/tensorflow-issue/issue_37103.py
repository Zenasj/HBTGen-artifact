# tf.random.uniform((B, 8), dtype=tf.float32) ← Input shape is (batch_size, 8) features as per code snippet with diabetes dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple feed-forward network for binary classification with logits output (no sigmoid)
        self.dense1 = tf.keras.layers.Dense(12, activation='relu', input_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(1)  # Logits output

        # Metrics that normally expect probabilities in [0,1]
        # We create "from_logits=True" adapted versions by internally applying sigmoid
        self.precision_metric = tf.keras.metrics.Precision(name='precision')
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        self.tp_metric = tf.keras.metrics.TruePositives(name='tp')
        self.fp_metric = tf.keras.metrics.FalsePositives(name='fp')
        self.tn_metric = tf.keras.metrics.TrueNegatives(name='tn')
        self.fn_metric = tf.keras.metrics.FalseNegatives(name='fn')
        self.recall_metric = tf.keras.metrics.Recall(name='recall')
        self.auc_metric = tf.keras.metrics.AUC(name='auc')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        logits = self.dense2(x)
        # For inference or loss calculation, return raw logits
        return logits

    def update_metrics(self, logits, labels):
        # Sigmoid to convert logits to probability [0,1]
        probs = tf.sigmoid(logits)

        # Update all metrics with probabilities (since base metrics expect in [0,1])
        self.precision_metric.update_state(labels, probs)
        self.accuracy_metric.update_state(labels, probs)
        self.tp_metric.update_state(labels, probs)
        self.fp_metric.update_state(labels, probs)
        self.tn_metric.update_state(labels, probs)
        self.fn_metric.update_state(labels, probs)
        self.recall_metric.update_state(labels, probs)
        self.auc_metric.update_state(labels, probs)

    def reset_metrics(self):
        self.precision_metric.reset_state()
        self.accuracy_metric.reset_state()
        self.tp_metric.reset_state()
        self.fp_metric.reset_state()
        self.tn_metric.reset_state()
        self.fn_metric.reset_state()
        self.recall_metric.reset_state()
        self.auc_metric.reset_state()

    def result_metrics(self):
        # Return a dictionary of all metric results
        return {
            'precision': self.precision_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'tp': self.tp_metric.result(),
            'fp': self.fp_metric.result(),
            'tn': self.tn_metric.result(),
            'fn': self.fn_metric.result(),
            'recall': self.recall_metric.result(),
            'auc': self.auc_metric.result(),
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor input matching 8 features expected by the model
    # Batch size chosen arbitrarily as 5 (same as example dataset rows)
    return tf.random.uniform((5, 8), dtype=tf.float32)

