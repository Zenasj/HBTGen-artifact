import torch
import torch.nn as nn

class UnsupervisedEventDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(UnsupervisedEventDetector, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.multihead_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim * 2, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        attention_output, _ = self.multihead_attention(encoded_output, encoded_output, encoded_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + encoded_output)
        decoded_output, _ = self.decoder(attention_output)
        output = self.output_layer(decoded_output)

        return output

# Create DataLoader
batch_size = 32
dataset = TensorDataset(input_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define model
input_dim = 6
hidden_dim = 64
num_layers = 3
num_heads = 4
model = UnsupervisedEventDetector(input_dim, hidden_dim, num_layers, num_heads).to(device)
# model.apply(weights_init)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Training
for epoch in range(3):
    model.train()
    for batch in data_loader:
        inputs = batch[0].to(device)
        reconstructed_output = model(inputs)
        
        loss = criterion(reconstructed_output, inputs)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step(loss)
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

tensor([[[0.2071, 0.0677, 0.0714, 0.2388, 0.2842, 0.2932],
         [0.2165, 0.0697, 0.0725, 0.2366, 0.2830, 0.2915],
         [0.2185, 0.0705, 0.0743, 0.2358, 0.2810, 0.2887],
         ...,
         [0.2538, 0.1079, 0.0916, 0.2612, 0.2853, 0.3129],
         [0.2567, 0.1091, 0.0926, 0.2627, 0.2873, 0.3147],
         [0.2603, 0.1097, 0.0965, 0.2640, 0.2868, 0.3153]],

        [[0.0802, 0.0685, 0.0217, 0.0692, 0.0271, 0.0522],
         [0.0822, 0.0714, 0.0219, 0.0693, 0.0266, 0.0520],
         [0.0834, 0.0736, 0.0221, 0.0696, 0.0276, 0.0519],
         ...,
         [0.1460, 0.0235, 0.0079, 0.0713, 0.0290, 0.0523],
         [0.1507, 0.0226, 0.0122, 0.0714, 0.0291, 0.0530],
         [0.1576, 0.0217, 0.0115, 0.0710, 0.0293, 0.0528]]], device='cuda:0')

tensor([[[ 0.1022,  0.0796,  0.0243,  0.0653,  0.0694, -0.0048],
         [ 0.1056,  0.0881,  0.0241,  0.0665,  0.0682, -0.0020],
         [ 0.1085,  0.0924,  0.0261,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1221,  0.0902,  0.0305,  0.0626,  0.0583,  0.0092],
         [ 0.1221,  0.0901,  0.0301,  0.0629,  0.0585,  0.0092],
         [ 0.1225,  0.0898,  0.0296,  0.0632,  0.0590,  0.0089]],

        [[ 0.1022,  0.0796,  0.0243,  0.0653,  0.0694, -0.0048],
         [ 0.1056,  0.0881,  0.0241,  0.0665,  0.0682, -0.0020],
         [ 0.1084,  0.0924,  0.0261,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1221,  0.0901,  0.0305,  0.0625,  0.0584,  0.0092],
         [ 0.1221,  0.0901,  0.0301,  0.0628,  0.0585,  0.0091],
         [ 0.1225,  0.0898,  0.0296,  0.0631,  0.0590,  0.0089]],

        [[ 0.1022,  0.0796,  0.0243,  0.0653,  0.0694, -0.0048],
         [ 0.1056,  0.0881,  0.0241,  0.0665,  0.0682, -0.0020],
         [ 0.1084,  0.0924,  0.0261,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1221,  0.0903,  0.0305,  0.0625,  0.0584,  0.0091],
         [ 0.1221,  0.0902,  0.0301,  0.0628,  0.0585,  0.0091],
         [ 0.1225,  0.0899,  0.0296,  0.0631,  0.0590,  0.0089]],

        ...,

        [[ 0.1022,  0.0796,  0.0243,  0.0652,  0.0694, -0.0048],
         [ 0.1057,  0.0882,  0.0240,  0.0665,  0.0682, -0.0020],
         [ 0.1086,  0.0925,  0.0260,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1221,  0.0900,  0.0302,  0.0625,  0.0586,  0.0091],
         [ 0.1222,  0.0899,  0.0298,  0.0628,  0.0587,  0.0091],
         [ 0.1226,  0.0896,  0.0293,  0.0631,  0.0592,  0.0088]],

        [[ 0.1022,  0.0796,  0.0243,  0.0652,  0.0694, -0.0048],
         [ 0.1057,  0.0881,  0.0240,  0.0665,  0.0682, -0.0020],
         [ 0.1085,  0.0924,  0.0259,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1221,  0.0901,  0.0302,  0.0625,  0.0584,  0.0092],
         [ 0.1222,  0.0900,  0.0299,  0.0628,  0.0586,  0.0091],
         [ 0.1225,  0.0897,  0.0293,  0.0631,  0.0590,  0.0089]],

        [[ 0.1022,  0.0796,  0.0243,  0.0653,  0.0694, -0.0048],
         [ 0.1056,  0.0881,  0.0240,  0.0665,  0.0682, -0.0020],
         [ 0.1085,  0.0924,  0.0260,  0.0656,  0.0677,  0.0007],
         ...,
         [ 0.1222,  0.0903,  0.0304,  0.0624,  0.0584,  0.0092],
         [ 0.1222,  0.0902,  0.0300,  0.0627,  0.0585,  0.0091],
         [ 0.1226,  0.0899,  0.0295,  0.0630,  0.0590,  0.0089]]],
       device='cuda:0')