import numpy as np

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
   

    vocab_len = len(word_to_index) + 1            #1193514      
    emb_matrix = np.zeros((vocab_len,embedding_dim))
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Definning a pre-trained Embedding layer
    embedding_layer = layers.Embedding(
                        vocab_len,
                        embedding_dim,
                        trainable = False
                        )

    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def sentiment_model(input_shape, word_to_vec_map, word_to_index):


    sentence_indices =layers.Input(shape=input_shape, dtype='float32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)   

    x = layers.LSTM(128)(embeddings)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(2, activation="sigmoid", name="predictions")(x)
    
    # Create Model instance which converts sentence_indices into X.
    model = keras.Model(inputs=sentence_indices,outputs=predictions)   
    return model

def sentences_to_indices(X, word_to_index, max_len):

    X_indices = np.zeros((m,max_len))
    
    # Assign indices to words
    for i,sentence in enumerate(X):        
        sentence_words = sentence.lower().split()
        for j,word in enumerate(sentence_words):
            X_indices[i, j] = word_to_index[word]
    return X_indices

X_train_indices = sentences_to_indices(X_train, word_to_index, max_features)
Y_train_OH = to_categorical(Y_train)
model.fit(X_train_indices, Y_train_OH, epochs = 10, batch_size = 32)