import onnx
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3"
feature_extractor = transformers.WhisperFeatureExtractor(feature_size=128)
device = "cpu"
batch = 4
onnx_path = "/opt/dev/whisper/whisper3.onnx"

real_input = {
    "input_features": torch.randn(
        (
            batch,
            feature_extractor.feature_size,
            feature_extractor.nb_max_frames,
        )
    ),
    "decoder_input_ids": torch.tensor([[1, 1]]) * 8001,
    "return_dict": False,
}
with torch.onnx.enable_fake_mode() as ctx:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, low_cpu_mem_usage=False, use_safetensors=False
    )
    input = {
        "input_features": torch.randn(
            (
                batch,
                feature_extractor.feature_size,
                feature_extractor.nb_max_frames,
            )
        ),
        "decoder_input_ids": torch.tensor([[1, 1]]) * 8001,
        "return_dict": False,
    }

export_options = torch.onnx.ExportOptions(fake_context=ctx)
onnx_program = torch.onnx.dynamo_export(model, **input, export_options=export_options)
onnx_program.save(onnx_path)
onnx.checker.check_model(onnx_path, full_check=True)