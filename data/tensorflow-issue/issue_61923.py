# ------------------ Encoder ------------------
encoder_model = Model(inputs=all_inputs, outputs=latent_space, name="encoder")
# ... irrelevant code 

# ------------------ Decoder ------------------
# Output Layers
numeric_output = Dense(
    self.numeric_dim, activation="linear", name="numeric_output")(decoder2)
binary_output = Dense(
    self.binary_dim, activation="sigmoid", name="binary_output")(decoder2)

decoder_output = [numeric_output] + [binary_output]
decoder_model = Model(inputs=latent_input, outputs=decoder_output, name="decoder")

# ------------------ Autoencoder ------------------
autoencoder_output = decoder_model(encoder_model(all_inputs))
autoencoder = Model(inputs=all_inputs, outputs=pass_through_layers, name="autoencoder")

# This will not work:
losses = {
            "numeric_output": "mse",
            "binary_output": "binary_crossentropy"
}
autoencoder.compile(optimizer=Adam(learning_rate=lr), loss=losses)