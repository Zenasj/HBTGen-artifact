import math
import tensorflow as tf

class Perplexity(keras_nlp.metrics.Perplexity):
    """Hack to make result() work with MirroredStrategy and mixed_precision"""

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        perplexity_score = tf.exp(
            tf.math.divide_no_nan(self._aggregate_crossentropy, self._number_of_samples)
        )
        return perplexity_score