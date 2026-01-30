import tensorflow as tf

def get_model_rnn():
    inputs = layers.Input(shape=(maxlen,))
    embedding = layers.Embedding(vocab_size, 128,  trainable=True)
    title_embed = embedding(inputs)
    title_ids_mask = layers.Masking(mask_value=0, name='mask')(title_embed)
    title_gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))(title_ids_mask)
    outputs = layers.Dense(1, activation='sigmoid', name='mlp2')(title_gru) 
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile("Adam", "binary_crossentropy", metrics=["binary_accuracy"])
    return model 
model = get_model_rnn()
model.save('./model_rnn')

converter = tf.lite.TFLiteConverter.from_saved_model('./model_rnn')
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT] # this caused the error
tflite_quant_model = converter.convert()
open("model_rnn.tflite", "wb").write(tflite_quant_model)

interpreter = tf.lite.Interpreter(model_path="./model_rnn.tflite")

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]