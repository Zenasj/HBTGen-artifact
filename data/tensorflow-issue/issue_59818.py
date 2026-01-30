from tensorflow import keras

from keras_cv.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
import tensorflow as tf 

signature_dict = {
    "tokens": tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name="tokens"),
}

def text_encoder_exporter(model: tf.keras.Model):
    BATCH_SIZE = 3
    MAX_PROMPT_LENGTH = 77
    POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
    UNCONDITIONAL_TOKENS = tf.convert_to_tensor([_UNCONDITIONAL_TOKENS], dtype=tf.int32)

    @tf.function(input_signature=[signature_dict])
    def serving_fn(inputs):
        # context
        encoded_text = model([inputs["tokens"], POS_IDS], training=False)
        encoded_text = tf.squeeze(encoded_text)

        if tf.rank(encoded_text) == 2:
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), BATCH_SIZE, axis=0
            )

        # unconditional context
        unconditional_context = model([UNCONDITIONAL_TOKENS, POS_IDS], training=False)

        unconditional_context = tf.repeat(unconditional_context, BATCH_SIZE, axis=0)
        return {"context": encoded_text, "unconditional_context": unconditional_context}

    return serving_fn

tf.saved_model.save(
    text_encoder,
    "./text_encoder/1/",
    signatures={"serving_default": text_encoder_exporter(text_encoder)},
)

from tensorflow.python.saved_model import tag_constants

batch_size = 3
saved_model_loaded = tf.saved_model.load(
    "./text_encoder/1/", tags=[tag_constants.SERVING]
)
text_encoder_predict_fn = saved_model_loaded.signatures["serving_default"]
# Raises error
xla_text_encoder_predict_fn = tf.function(text_encoder_predict_fn, jit_compile=True)
xla_text_encoder_predict_fn(
    tokens=tf.ones((batch_size, MAX_PROMPT_LENGTH), tf.int32)
).keys()