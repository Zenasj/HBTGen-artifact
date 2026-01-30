import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch

model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

signal, fs = torchaudio.load('shortTeaching2.wav')
print( signal )
print( signal.shape )
#exit( 0 ) 

# Print output shape
embeddings = model.encode_batch(signal)
#print( embeddings )
#print( embeddings.shape )
#exit( 0 ) 

# Create dummy input
symbolic_names = {0: "batch_size", 1: "max_seq_len"}
x = torch.randn( 1, 1920000 )
# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "embeddings.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  verbose=False,
                  input_names = ['signal'],   # the model's input names
                  output_names = ['embeddings'], # the model's output names
                  dynamic_axes={'signal' : symbolic_names,    # variable length axes
                                })