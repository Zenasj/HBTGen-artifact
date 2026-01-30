import torch

model_path = 'best_model.pth.tar'
model_config = 'config.json'

use_cuda = False

config = load_config(model_config)
ap = AudioProcessor(**config.audio)

input_adapter = lambda sen: text_to_sequence(sen, [config.text_cleaner])
input_size = len(symbols)

test_model = Tacotron(input_size, config.embedding_size, ap.num_freq, ap.num_mels, config.r)
if use_cuda:
    cp = torch.load(model_path)
else:
    cp = torch.load(model_path, map_location=lambda storage, loc: storage)
test_model.load_state_dict(cp['model'])
if use_cuda:
    test_model.cuda()
test_model.eval()

dummpy_input = torch.ones(1, MAX_LEN, dtype=torch.long)

torch.onnx.export(test_model, args=dummpy_input, f='test.onnx', verbose=True)

decoder_rnn_hiddens = [
             self.decoder_rnn_inits(inputs.data.new_tensor([idx]*B).long())
             for idx in range(len(self.decoder_rnns))
]