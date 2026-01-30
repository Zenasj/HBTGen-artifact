import tensorflow as tf
from tensorflow import keras

class Dora_A(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Dora_A, self).__init__(**kwargs)
        self.bioRoberta = TFRobertaModel.from_pretrained('allenai/biomed_roberta_base', from_pt=True)

    @tf.function
    def call(self, inputIds):
        queryInputs, passageInputs = inputIds

        Q_outputs = self.bioRoberta(queryInputs)[0]
        P_outputs = self.bioRoberta(passageInputs)[0]

        dotProductMatrix = tf.linalg.matmul(Q_outputs, P_outputs, transpose_b=True, name='mm')

        return dotProductMatrix

@tf.function
def loss_fn(_, probs):
    '''
        1. Every sample is its own positive, and  the rest of the
            elements in the batch are its negative.
        2. Each TPU core gets 1/8 * global_batch_size elements, hence
            compute shape dynamically.
        3. Dataset produces dummy labels to make sure the loss_fn matches
            the loss signature of keras, actual labels are computed inside this
            function.
        4. Inputs are logits, for better numerical stability.
    '''
    bs = tf.shape(probs)[0]
    labels = tf.eye(bs, bs)
    return tf.losses.categorical_crossentropy(labels,
                                              probs,
                                              from_logits=True)

class Dora_A(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Dora_A, self).__init__(**kwargs)
        self.bioRoberta = TFRobertaModel.from_pretrained('allenai/biomed_roberta_base', from_pt=True)

    @tf.function
    def call(self, inputIds):
        queryInputs, passageInputs = inputIds

        Q_outputs = self.bioRoberta(queryInputs)[0]
        P_outputs = self.bioRoberta(passageInputs)[0]

        dotProductMatrix = tf.matmul(Q_outputs, P_outputs, transpose_b=True, name='mm')

        return dotProductMatrix

# We allow three types of functions/objects passed into TPUStrategy
# run in eager mode:
#   1. a user annotated tf.function
#   2. a ConcreteFunction, this is mostly what you get from loading a saved
#      model.
#   3. a callable object and the `__call__` method itself is a tf.function.
#
# Otherwise we return an error, because we don't support eagerly running
# run in TPUStrategy.