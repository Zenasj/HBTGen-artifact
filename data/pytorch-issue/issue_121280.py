import torch.nn as nn

import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.latent_dim = 256
        self.num_heads = 4
        self.ff_size=1024
        self.dropout=0.1
        self.activation="gelu"
        self.num_layers = 4

        root_seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)

        self.root_seqTransEncoder = torch.nn.TransformerEncoder(root_seqTransEncoderLayer,
                                                          num_layers=self.num_layers)
        

    def forward(self, inputs):
        xseq = inputs[0]
        xseq = xseq.detach().requires_grad_()
        with torch.enable_grad():

            output = self.root_seqTransEncoder(xseq)
            loss = torch.sqrt(output).sum()

            return torch.autograd.grad([loss], [xseq])[0]

mdl = Model()
for p in mdl.parameters():
    p.requires_grad_(False)

print("export model")
torch.onnx.export(
    Model(),
    [torch.randn([20, 2, 256]) ** 2],
    "modelthing.onnx",
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)