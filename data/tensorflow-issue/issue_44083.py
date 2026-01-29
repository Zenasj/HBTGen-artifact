# tf.random.uniform((B, maxlen), dtype=tf.int32) for each of the 6 inputs - input_ids, token_type_ids, attention_mask for Document and Question

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, maxlen):
        super().__init__()
        # Load the pre-trained BERT layer from TF Hub, trainable
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
        
        # Dense layers in sequence as described, final output with 2 units (for yes/no)
        self.concat = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(256, activation='relu')
        self.dense4 = tf.keras.layers.Dense(128, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.out_layer = tf.keras.layers.Dense(2)  # No activation, logits output
    
    def call(self, inputs, training=False):
        # inputs expected as list of 6 inputs in order:
        # [input_ids_Document, token_type_ids_Document, attention_mask_Document,
        #  input_ids_Question, token_type_ids_Question, attention_mask_Question]
        
        input_ids_doc, token_type_ids_doc, attention_mask_doc, \
        input_ids_q, token_type_ids_q, attention_mask_q = inputs
        
        # BERT expects inputs as list/tensor: [input_ids, token_type_ids, attention_mask]
        bert_inputs_doc = [input_ids_doc, token_type_ids_doc, attention_mask_doc]
        bert_inputs_q = [input_ids_q, token_type_ids_q, attention_mask_q]
        
        # BERT layer outputs tuple: (pooled_output, sequence_output)
        bert_output_doc = self.bert_layer(bert_inputs_doc)[0]  # pooled_output shape (B, 768)
        bert_output_q = self.bert_layer(bert_inputs_q)[0]      # pooled_output shape (B, 768)
        
        # Concatenate pooled outputs from Document and Question
        concat_out = self.concat([bert_output_doc, bert_output_q])  # shape (B, 1536)
        
        x = self.dense1(concat_out)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.flatten(x)
        logits = self.out_layer(x)  # shape (B, 2)
        return logits

def my_model_function():
    # Based on the input data specs, maxlen appears to be the sequence length (e.g. 512)
    # vocab_size is used only for tokenizer, not for BERT layer inputs
    # We provide some reasonable defaults for illustration; user should override as needed
    maxlen = 512
    vocab_size = 30522  # Standard BERT vocab size
    
    return MyModel(vocab_size=vocab_size, maxlen=maxlen)

def GetInput():
    # Returns a list of 6 tensors simulating batch input to MyModel:
    # [input_ids_Document, token_type_ids_Document, attention_mask_Document,
    #  input_ids_Question, token_type_ids_Question, attention_mask_Question]
    
    batch_size = 2  # example batch size
    maxlen = 512    # example sequence length
    
    # All inputs are int32 tensor with shape (batch_size, maxlen)
    # input_ids: integers in [0, vocab_size), token_type_ids: 0 or 1
    # attention_mask: 0 or 1
    
    input_ids_doc = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=30522, dtype=tf.int32)
    token_type_ids_doc = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=2, dtype=tf.int32)
    attention_mask_doc = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=2, dtype=tf.int32)
    
    input_ids_q = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=30522, dtype=tf.int32)
    token_type_ids_q = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=2, dtype=tf.int32)
    attention_mask_q = tf.random.uniform(
        shape=(batch_size, maxlen), minval=0, maxval=2, dtype=tf.int32)
    
    return [input_ids_doc, token_type_ids_doc, attention_mask_doc,
            input_ids_q, token_type_ids_q, attention_mask_q]

