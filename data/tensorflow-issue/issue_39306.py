import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model_spec = BertClassifierModelSpec(uri='https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1')

train_data = TextClassifierDataLoader.from_folder(os.path.join(data_path, 'train'), model_spec=model_spec, class_labels=['pos', 'neg'])

test_data = TextClassifierDataLoader.from_folder(os.path.join(data_path, 'test'), model_spec=model_spec, is_training=False, shuffle=False)

model = text_classifier.create(train_data, model_spec=model_spec, epochs=1)

if is_tf2:
    bert_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable,
                                signature='tokens', signature_outputs_as_dict = True)
    pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                       dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                    dtype=tf.int32,
                                    name="segment_ids")

litebert = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_lite_base/1",
                          signature="tokens",
                          signature_outputs_as_dict=True,
                          name="albert_lite")

pooled_output = litebert(dict(input_ids=input_word_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))["pooled_output"]

output = tf.keras.layers.Dropout(rate=0.0001)(pooled_output)

output = tf.keras.layers.Dense(
    2,
    name='output',
    dtype=tf.float32)(output)

model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=output)

model.compile()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with tf.io.gfile.GFile(os.path.join(path, "temp.tflite"), 'wb') as f:
    f.write(tflite_model)