# tf.random.uniform((B, 2), dtype=tf.string) ← The input model expects a dict with keys 'question' and 'answer', each a string tensor

import tensorflow as tf
import tensorflow_hub as hub

# Using the official BERT base uncased model and preprocess from TF Hub from the issue context
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=2, seq_length=128, **kwargs):
        super().__init__(**kwargs)
        # Build the preprocessing model that takes sentence features ("question" and "answer")
        self.seq_length = seq_length
        self.sentence_features = ['question', 'answer']

        # Load preprocessing hub module
        self.bert_preprocess = hub.load(tfhub_handle_preprocess)
        # Keras layers wrapping hub modules for tokenization and packing
        self.tokenizer = hub.KerasLayer(self.bert_preprocess.tokenize, name='tokenizer')
        self.packer = hub.KerasLayer(self.bert_preprocess.bert_pack_inputs,
                                     arguments=dict(seq_length=seq_length),
                                     name='packer')

        # Load encoder layer
        self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='encoder')

        # Classifier head components
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')  # From logits used with SparseCategoricalCrossentropy

    def call(self, inputs, training=False):
        # inputs is expected as a dict: {'question': string tensor, 'answer': string tensor}
        # Tokenize each input string feature
        segments = []
        for ft in self.sentence_features:
            # inputs[ft] shape: (B,), dtype=tf.string
            tokens = self.tokenizer(inputs[ft])
            segments.append(tokens)

        # Pack tokenized inputs into BERT input format
        bert_inputs = self.packer(segments)
        # Encode using BERT
        encoder_outputs = self.encoder(bert_inputs)
        pooled_output = encoder_outputs['pooled_output']  # shape (B, 768)
        x = self.dropout(pooled_output, training=training)
        logits = self.classifier(x)
        return logits


def my_model_function():
    # Return an instance of MyModel with default parameters
    return MyModel()


def GetInput():
    # Generate a random batch of inputs mimicking the expected input dictionary with string tensors.
    # Since this is a test/demo, create a batch of 4 samples with dummy strings
    batch_size = 4
    # Create dummy string inputs for 'question' and 'answer' keys
    question_tensor = tf.constant(
        ['What is TensorFlow?', 'How to use BERT?', 'Explain QNLI task.', 'Is TPU faster?'], dtype=tf.string)
    answer_tensor = tf.constant(
        ['TensorFlow is a machine learning framework.',
         'BERT is used for NLP tasks.',
         'QNLI is a question-answering task.',
         'TPUs provide fast training.'], dtype=tf.string)
    # Return dictionary matching the expected input signature
    return {'question': question_tensor, 'answer': answer_tensor}

