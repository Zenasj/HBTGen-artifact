import torch.nn as nn

import torch
import torchaudio

class ConformerSpeechRecognizer(torch.nn.Module):
    def __init__(self,
                 kernel_size,
                 ffn_dim: int,
                 feature_vector_size: int,
                 hidden_layer_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 depthwise_conv_kernel_size: int,
                 vocabulary_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.cnn_ = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=feature_vector_size),
            torch.nn.Conv1d(
                in_channels=feature_vector_size,
                out_channels=hidden_layer_size,
                bias=False,
                kernel_size=(kernel_size,),
                padding='same'
            ),
            torch.nn.BatchNorm1d(num_features=hidden_layer_size)
        )
        self.conformer_ = torchaudio.models.Conformer(
            input_dim=hidden_layer_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )
        self.proba_appoximator_ = torch.nn.Linear(
            in_features=hidden_layer_size,
            out_features=vocabulary_size
        )
    def forward(self, inputs: torch.Tensor, input_lenghts: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.gelu(self.cnn_(inputs.permute(0, 2, 1)))
        hidden_states, _ = self.conformer_(hidden_states.permute(0, 2, 1), input_lenghts)
        output_logits = self.proba_appoximator_(hidden_states)
        return output_logits

# my_module = ConformerSpeechRecognizer(
#                 hidden_layer_size=512,
#                 ffn_dim=64,
#                 num_heads=8,
#                 kernel_size=9,
#                 feature_vector_size=13,
#                 num_layers=6,
#                 dropout=0.41130458168952116,
#                 depthwise_conv_kernel_size=9
#                 vocabulary_size=37)

import torch
from conformer_model import ConformerSpeechRecognizer

model = torch.load("model.pt")
torch.onnx.export(model.cpu(),
                  f="../model.onnx",
                  input_names=["inputs", "input_lenghts"],
                  output_names=["logits"],
                  args=({ "inputs": torch.ones(1, 1, 13), "input_lenghts": torch.ones(1, dtype=torch.long) }),
                  dynamic_axes={
                    "inputs": { 0: "batch_count", 1: "batch_item_length" },
                    "input_lenghts": { 0: "batch_count" },
                    "logits": { 0: "batch_count", 1: "batch_item_length" }
                  },
                  verbose=True
                )

import torch
import torchaudio
import conformer_model
from transformers import Wav2Vec2CTCTokenizer
import onnxruntime as ort
import numpy

testDir = "test conformer/";
testFiles = ["1_0.wav", "1_1.wav", "1_2.wav"];  

def loadTestFileFeatureMatrix(fileName):
  waveform, sample_rate = torchaudio.load(testDir + fileName, normalize=True)
  transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
      'n_mels': 64,
      'n_fft': 512,
      'win_length': 400,
      'hop_length': 320,
      'center': False,
      'power': 2,
      'f_min': 200,
      'f_max': 5000
    }
  )

  mfcc = transform(waveform)

  numpy_array = mfcc[0].numpy()
  numpy.savetxt(testDir + fileName.rsplit('.', 1)[0] + "_spectral_array.txt", numpy_array)

  return mfcc


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("conformer")

feature_lengths = []
features = []

for file in testFiles:
  with torch.no_grad():
    mfcc = loadTestFileFeatureMatrix(file)
    new_feature_matrix = mfcc
    new_feature_matrix = torch.moveaxis(mfcc, 1, 2)
  feature_lengths.append(new_feature_matrix.cpu().size()[1])
  features.append(new_feature_matrix[0])

features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
feature_lengths = torch.from_numpy(numpy.array(feature_lengths, dtype=numpy.int32)).to(torch.long)

print(features.shape)
print(feature_lengths.shape)

session = ort.InferenceSession("conformer-onnx/model.onnx")

ortvalue = ort.OrtValue.ortvalue_from_numpy(features.numpy())
ortvalue2 = ort.OrtValue.ortvalue_from_numpy(feature_lengths.numpy())

input_name = session.get_inputs()[0].name
input2_name = session.get_inputs()[1].name
label_name = session.get_outputs()[0].name

print(input_name)
print(input2_name)
print(label_name)

result = session.run([label_name], { input_name: ortvalue, input2_name: ortvalue2 })