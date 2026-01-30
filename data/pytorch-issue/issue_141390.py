import torch
import torchaudio

bundle = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
bundle.eval()
feature_extractor = bundle.feature_extractor
encoder = bundle.encoder

example_wav_features = torch.randn(1, 512)
example_length = torch.rand(1)
example_features = feature_extractor(example_wav_features, example_length)
exported_encoder_model = torch.export.export(encoder, example_features)
print(exported_encoder_model)