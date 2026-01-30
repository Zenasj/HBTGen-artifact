import numpy as np

# Padding
X = [[word2idx[w[0]] for w in s] for s in quran_sentences]
X = pad_sequences(maxlen=max_length, sequences=X, padding="post",value=word2idx["PAD"])

y = [[tag2idx[w[1]] for w in s] for s in quran_sentences]
y = pad_sequences(maxlen=max_length, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
hidden_state_encoder_size = 100

# hidden_state_decoder_size = 200

batch_size = 64

training_epoch = 200

embedding_size = 80

dropout_rate = 0.5

# Model
# Input
inputs = Input(shape=(max_length,), name="Input")

# Embedding
# Output = (batch_size, input_length, output_dim)
embed = Embedding(input_dim=n_words+1,
                  output_dim=embedding_size,
                  input_length=max_length,
                  name="Embedding")(inputs)

# Bi-LSTM
# Output = (batch_size, steps, features)
encoder = Bidirectional(LSTM(units=hidden_state_encoder_size,
                             return_sequences=True,
                             dropout=dropout_rate,
                             name="LSTM"),
                        name="Bi-LSTM")

hidden_states = encoder(embed)

# Average
# Output = (batch_size, features)
average = GlobalAveragePooling1D(name="Average")(hidden_states)

# Outputs
outputs = Dense(n_tags,
                activation="softmax",
                name="Output")(average)

model = Model(inputs, outputs, name="Sequence Chunking")

# Compile & Train
# Compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

# run training
model.fit(X_tr, np.array(y_tr),
          batch_size=batch_size,
          epochs=training_epoch,
          validation_split=0.1)