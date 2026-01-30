import torch
import torch.nn as nn
import torch.onnx

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using ' + device)

    input_dim = 2
    hidden_dim = 4
    batch_size = 8
    x = torch.randn((batch_size, input_dim), device=device)
    h0 = torch.randn((batch_size, hidden_dim), device=device)
    c0 = torch.randn((batch_size, hidden_dim), device=device)
    print("******input*******")
    print("x, h0, c0", x.size(), h0.size(), c0.size())
    print("")

    print("******net*******")
    net = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim).to(device)
    print(net)
    print("")

    print("******output*******")
    h1, c1 = net(x, (h0, c0))
    print("h1, c1", h1.size(), c1.size())
    print("")

    print("******onnx.export*******")
    torch.onnx.export(net, (x, (h0, c0)), "test.onnx", verbose=True,
                      input_names=['x', 'h0', 'c0'], output_names=['h1', 'c1'])
    print("")

# Config
USE_CPU = True
OPSET_VERSION = 11

# Load model
tacotron2 = load_taco2(args.tacotron2)
tacotron2.eval()

# Generate dummy data
sequences = torch.randint(low=0, high=148, size=(1, 50), dtype=torch.long)
sequence_lengths = torch.IntTensor([sequences.size(1)]).long()
dummy_input = (sequences, sequence_lengths)
if not USE_CPU:
    dummy_input = tuple(t.cuda() for t in dummy_input)

# Run it through
with torch.no_grad():
    tacotron2(*dummy_input)

# Export model
print('exporting...')
torch.onnx.export(tacotron2, dummy_input, args.output + "/" + "taco2.onnx",
                  opset_version=OPSET_VERSION,
                  do_constant_folding=True,
                  input_names=["sequences", "sequence_lengths"],
                  output_names=["mel_outputs_postnet", "mel_lengths", "alignments"],
                  dynamic_axes={"sequences": {1: "text_seq"}})
print('done!')