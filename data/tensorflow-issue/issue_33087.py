from tensorflow.keras import layers

encoder_inputs = Input(shape=(None, vocab_size))
decoder_inputs = Input(shape=(None, vocab_size))

encoder_bigru = Bidirectional(GRU(units, return_sequences=True, return_state=True, dropout=dropout))

encoder_out, encoder_fwd_state, encoder_back_state = encoder_bigru(encoder_inputs)
encoder_states = Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])

decoder_gru = GRU(units * 2, return_sequences=True, return_state=True, dropout=dropout)

decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_states)

attn_out = Attention()([decoder_out, encoder_out])
decoder_concat_input = Concatenate(axis=-1)([decoder_out, attn_out])

dense_time = TimeDistributed(Dense(vocab_size, activation="softmax"))

decoder_pred = dense_time(decoder_concat_input)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
model.compile(optimizer="adam", loss="categorical_crossentropy")

from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Concatenate, Bidirectional, GRU, TimeDistributed, Dense, Attention

callbacks = [
  CSVLogger(filename=PATH, separator=";", append=True),
  TensorBoard(log_dir=PATH, histogram_freq=10, profile_batch=0, write_graph=True, write_images=False, update_freq="epoch"),
  ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
  EarlyStopping(monitor="val_loss", min_delta=0.001, patience=40, restore_best_weights=True, verbose=1)
]

[one_hot_inputs, one_hot_decoders], one_hot_targets