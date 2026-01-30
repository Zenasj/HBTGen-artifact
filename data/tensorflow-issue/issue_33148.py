self.model = Sequential([
            Embedding(len(self.item_map), self.embed_dim, input_length = X.shape[1],mask_zeros=True),
            LSTM(self.lstm_out),
            Dense(len(self.item_map)-1),
        ])

self.model = Sequential([
            Embedding(len(self.item_map), self.embed_dim, input_length = X.shape[1]),
            Masking(mask_value=0),
            LSTM(self.lstm_out),
            Dense(len(self.item_map)-1),
        ])