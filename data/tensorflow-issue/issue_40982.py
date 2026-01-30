# Input and embeddings for words
word_in = Input(shape=(max_len, max_len_word,))

# Word level embedding
emb_word = TimeDistributed(
    Embedding(input_dim=(n_words + 2), 
        output_dim=200,
        input_length=max_len_word, 
        weights=[get_embedding_matrix(word_index, 
            embedding_path, embedding_dim)], 
        trainable=False,
        mask_zero=True
	)	
)(word_in)

# Word LSTM to get sent encodings by words
emb_sent = TimeDistributed(LSTM(units=32, return_sequences=False))(emb_word)

main_lstm = Bidirectional(LSTM(units=64, return_sequences=True))(emb_sent)
out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

model = Model([word_in], out)