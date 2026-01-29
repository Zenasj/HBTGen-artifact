# tf.random.uniform((B, None, None, C), dtype=tf.float32) ← Input shape here is (batch_size, sequence_length, embedding_dim)
# From the example, embedding_dim corresponds to config.hidden_size=768, but sequence length is dynamic (None)

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import AutoTokenizer, TFRobertaModel

# Utility and activation functions from the provided code

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """ Gaussian Error Linear Unit approximation. """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
}

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def cast_bool_to_primitive(bool_variable, default_tensor_to_true=False):
    """Cast possible boolean tensor or bool variable to python bool."""
    if tf.is_tensor(bool_variable):
        if hasattr(bool_variable, "numpy"):
            return bool(bool_variable.numpy())
        elif default_tensor_to_true:
            return True
    return bool_variable


# Configuration dictionary and AttrDict wrapper

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


configBase = {
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "type_vocab_size": 1,
  "vocab_size": 50265
}

config = AttrDict(configBase)


# -------------------
# Define two types of Transformer layers based on the code chunks: 
# TFBertLayer (original) and TFBertLayer2 (variant, with some name suffix changes and activation adjustments)
# All sub-layers for each are included as defined in chunks.
# -------------------

class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query_"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key_"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value_"
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (
            (context_layer, attention_probs) if cast_bool_to_primitive(output_attentions) is True else (context_layer,)
        )

        return outputs


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFBertSelfAttention(config, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask, output_attentions = inputs

        self_outputs = self.self_attention([input_tensor, attention_mask, head_mask, output_attentions], training=training)
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class TFBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFBertAttention(config, name="attention")
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        self.bert_output = TFBertOutput(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        attention_outputs = self.attention([hidden_states, attention_mask, head_mask, output_attentions], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


# Variant TFBertLayer2 with "2" suffixes in layer names and a tanh activation on intermediate layer

class TFBertSelfAttention2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query_2"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key_2"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value_2"
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (
            (context_layer, attention_probs) if cast_bool_to_primitive(output_attentions) is True else (context_layer,)
        )

        return outputs


class TFBertSelfOutput2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense2"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm2")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertAttention2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFBertSelfAttention2(config, name="self2")
        self.dense_output = TFBertSelfOutput2(config, name="output2")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask, output_attentions = inputs

        self_outputs = self.self_attention([input_tensor, attention_mask, head_mask, output_attentions], training=training)
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class TFBertIntermediate2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense2"
        )
        # The original code sets this to tanh later manually on the instance
        # Here we initialize with gelu by default, will override in model creation
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TFBertOutput2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense2"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm2")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFBertLayer2(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFBertAttention2(config, name="attention2")
        self.intermediate = TFBertIntermediate2(config, name="intermediate2")
        self.bert_output = TFBertOutput2(config, name="output2")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        attention_outputs = self.attention([hidden_states, attention_mask, head_mask, output_attentions], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


# -------------------

# Function to get two transformer layers (one TFBertLayer and one TFBertLayer2) with weights copied from pretrained model
def get_2_transformerLayerP(numb):
    """
    Creates two transformer layers from a pretrained biomed_roberta_base model:
    - TFBertLayer named layer_._{11+numb}
    - TFBertLayer2 named layer_._{12+numb}
    Weights for layer 10 and 11 from pretrained are copied to these layers.
    Also sets the intermediate activation of the TFBertLayer2 to tanh as per the original code.
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/biomed_roberta_base')
    inputt = tokenizer.encode('This is a sentence', return_tensors='tf')
    tempModel = TFRobertaModel.from_pretrained('allenai/biomed_roberta_base', from_pt=True)
    outt = tempModel(inputt)[0]

    t_layer11 = TFBertLayer(config, name="layer_._{}".format(11+numb))
    t_layer12 = TFBertLayer2(config, name="layer_._{}".format(12+numb))

    # Call once to build variables
    t_layer11((outt, None, None, None))
    t_layer12((outt, None, None, None))

    # Copy weights from pretrained roberta layers 10 and 11 (0-based indexing)
    t_layer11.set_weights(tempModel.layers[0].encoder.layer[10].get_weights())
    t_layer12.set_weights(tempModel.layers[0].encoder.layer[11].get_weights())

    # Override intermediate activation on t_layer12 to tanh (as per code comment)
    t_layer12.intermediate.intermediate_act_fn = tf.keras.activations.tanh

    del tokenizer
    del tempModel

    return t_layer11, t_layer12


# The main composite model class that contains two transformer layers as submodules and runs them sequentially
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Get the two transformer layers (hardcoded number 6 as per original code)
        self.layer1, self.layer2 = get_2_transformerLayerP(6)

    def call(self, inputs, training=False):
        # Inputs is tensor of shape [batch_size, seq_len, hidden_dim=768]
        x = inputs
        # Run first transformer layer; pass None for attention_mask, head_mask, output_attentions as in original
        out1 = self.layer1((x, None, None, None), training=training)[0]
        # Run second transformer layer on output of first
        out2 = self.layer2((out1, None, None, None), training=training)[0]
        return out2


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random tensor with dynamic batch size 2, sequence length 16, hidden size 768
    # dtype is float32 as used in transformer embeddings
    batch_size = 2
    seq_length = 16
    hidden_size = config.hidden_size  # 768
    return tf.random.uniform((batch_size, seq_length, hidden_size), dtype=tf.float32)

