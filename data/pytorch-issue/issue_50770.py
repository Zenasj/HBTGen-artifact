import torch

model = LSTMEncoderDecoder(n_features, emb_size)

model.load_state_dict(
    torch.load(
        f'data/models/ae_lstm_mse_sum_{window}d_{factor}f.pt', # The location of the saved model
        map_location=torch.device(device)                                        # For loading the model onto a CPU only device.
    ) 
)