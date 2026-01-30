import os
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    DynamicCache,
)

model_id="google/paligemma2-3b-pt-224"
# model_id="google/paligemma2-3b-ft-docci-448"
# model_id="google/paligemma2-3b-pt-448"
# model_id="google/paligemma2-3b-pt-896"

def new_len(self: torch.Tensor):
    return self.shape[0]

torch.Tensor.__len__ = new_len


class VisionEncoder(nn.Module):
  def __init__(self, paligemma_model):
    super().__init__()
    self.config = paligemma_model.config
    self.vision_tower = paligemma_model.vision_tower
    self.multi_modal_projector = paligemma_model.multi_modal_projector

  def forward(self, pixel_values: torch.FloatTensor):
      """
      Obtains image last hidden states from the vision tower and apply multimodal projection.

      Args:
          pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
              The tensors corresponding to the input images.
      Returns:
          image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
      """
      image_outputs = self.vision_tower(pixel_values)
      selected_image_feature = image_outputs.last_hidden_state
      image_features = self.multi_modal_projector(selected_image_feature)
      image_features = image_features / (self.config.text_config.hidden_size**0.5)
      return image_features


class PatchedPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    def forward(self, *args):
        inputs_embeds, position_ids, *past_key_values_args = args
        config = model.config.text_config

        # Convert past_key_values list to DynamicCache
        if len(past_key_values_args) == 0:
            past_key_values = None
        else:
            past_key_values = DynamicCache(config.num_hidden_layers)
            for i in range(config.num_hidden_layers):
                key = past_key_values_args.pop(0)
                value = past_key_values_args.pop(0)
                past_key_values.update(key_states=key, value_states=value, layer_idx=i)


        batch_size = inputs_embeds.shape[0]

        o = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            # Create a 4D attention mask of all zeros (attend to everything)
            attention_mask=torch.zeros(
                batch_size,
                1, # num_attention_heads (1 -> expand to num_attention_heads)
                1, # sequence_length (1 -> expand to sequence_length)
                1, # total_sequence_length (1 -> expand to total_sequence_length)
                dtype=torch.float32,
            ),
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        flattened_past_key_values_outputs = {
            "logits": o.logits,
        }
        output_past_key_values: DynamicCache = o.past_key_values
        for i, (key, value) in enumerate(
            zip(output_past_key_values.key_cache, output_past_key_values.value_cache)
        ):
            flattened_past_key_values_outputs[f"present.{i}.key"] = key
            flattened_past_key_values_outputs[f"present.{i}.value"] = value

        return flattened_past_key_values_outputs


# Constants
OUTPUT_FOLDER = os.path.join("output", model_id)
TEXT_MODEL_NAME = "decoder_model_merged.onnx"
VISION_MODEL_NAME = "vision_encoder.onnx"
EMBED_MODEL_NAME = "embed_tokens.onnx"
TEMP_MODEL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "temp")
FINAL_MODEL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "onnx")


# Load model and processor
model = PatchedPaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
).eval()
vision_model = VisionEncoder(model)
embed_layer = model.language_model.model.embed_tokens

processor = AutoProcessor.from_pretrained(model_id)

# Save model configs and processor
model.config.save_pretrained(OUTPUT_FOLDER)
model.generation_config.save_pretrained(OUTPUT_FOLDER)
processor.save_pretrained(OUTPUT_FOLDER)
os.makedirs(TEMP_MODEL_OUTPUT_FOLDER, exist_ok=True)


# Configuration values
## Text model
text_config = model.config.text_config
num_attention_heads = text_config.num_attention_heads
num_key_value_heads = text_config.num_key_value_heads
head_dim = text_config.head_dim
num_layers = text_config.num_hidden_layers
hidden_size = text_config.hidden_size

# Dummy input sizes
batch_size = 2
sequence_length = 32
past_sequence_length = 8

## Text inputs
dummy_past_key_values_kwargs = {
    f"past_key_values.{i}.{key}": torch.zeros(
        batch_size,
        num_key_value_heads,
        past_sequence_length,
        head_dim,
        dtype=torch.float32,
    )
    for i in range(num_layers)
    for key in ["key", "value"]
}
inputs_embeds = torch.randn(
    (batch_size, sequence_length, hidden_size),
)

total_sequence_length = sequence_length + past_sequence_length
position_ids = torch.arange(1, sequence_length + 1, dtype=torch.int64).expand(batch_size, sequence_length)

text_inputs = dict(
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    **dummy_past_key_values_kwargs,
)
text_inputs_positional = tuple(text_inputs.values())
text_outputs = model.forward(*text_inputs_positional)  # Test forward pass

## Vision inputs
size = processor.image_processor.size
w, h = size['width'], size['height']
pixel_values = torch.randn(2, 3, h, w, requires_grad=True)
vision_inputs = dict(pixel_values=pixel_values)
vision_inputs_positional = tuple(vision_inputs.values())
vision_outputs = vision_model.forward(*vision_inputs_positional)  # Test forward pass



# ONNX Exports
from torch.onnx._globals import GLOBALS
GLOBALS.onnx_shape_inference = False # Bug in pytorch

## Text model
TEXT_MODEL_OUTPUT_PATH=os.path.join(TEMP_MODEL_OUTPUT_FOLDER, TEXT_MODEL_NAME)
torch.onnx.export(
    model,
    args=text_inputs_positional,
    f=TEXT_MODEL_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=list(text_inputs.keys()),
    output_names=["logits"]
    + [f"present.{i}.{key}" for i in range(num_layers) for key in ["key", "value"]],
    dynamic_axes={
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        **{
            f"past_key_values.{i}.{key}": {0: "batch_size", 2: "past_sequence_length"}
            for i in range(num_layers)
            for key in ["key", "value"]
        },
        "logits": {0: "batch_size", 1: "sequence_length"},
        **{
            f"present.{i}.{key}": {0: "batch_size", 2: "total_sequence_length"}
            for i in range(num_layers)
            for key in ["key", "value"]
        },
    },

)

## Vision model
VISION_MODEL_OUTPUT_PATH = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, VISION_MODEL_NAME)
torch.onnx.export(
    vision_model,
    args=vision_inputs_positional,
    f=VISION_MODEL_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['pixel_values'],
    output_names=['image_features'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'image_features': {0: 'batch_size'}
    },
)

input_ids = torch.randint(0, embed_layer.num_embeddings, (batch_size, sequence_length))

## Embedding model
EMBED_MODEL_OUTPUT_PATH = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, EMBED_MODEL_NAME)
torch.onnx.export(
    embed_layer,
    args=(input_ids,),
    f=EMBED_MODEL_OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['inputs_embeds'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'}
    },
)


# Post-processing
import onnx
import onnxslim
from optimum.onnx.graph_transformations import check_and_save_model

os.makedirs(FINAL_MODEL_OUTPUT_FOLDER, exist_ok=True)
for name in (TEXT_MODEL_NAME, VISION_MODEL_NAME, EMBED_MODEL_NAME):
    temp_model_path = os.path.join(TEMP_MODEL_OUTPUT_FOLDER, name)

    onnx.shape_inference.infer_shapes_path(temp_model_path, check_type=True, strict_mode=True)

    ## Attempt to optimize the model with onnxslim
    try:
        onnx_model = onnxslim.slim(temp_model_path)
    except Exception as e:
        print(f"Failed to slim {temp_model_path}: {e}")
        onnx_model = onnx.load(temp_model_path)

    ## Save model
    final_model_path = os.path.join(FINAL_MODEL_OUTPUT_FOLDER, name)
    check_and_save_model(onnx_model, final_model_path)


# Minify tokenizer.json
import json
tokenizer_path = os.path.join(OUTPUT_FOLDER, "tokenizer.json")
with open(tokenizer_path, "r") as f:
    tokenizer = json.load(f)
with open(tokenizer_path, "w") as f:
    json.dump(tokenizer, f) # No need for indenting

# Add head_dim and num_image_tokens to config.json
config_path = os.path.join(OUTPUT_FOLDER, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config["text_config"]["head_dim"] = head_dim
config["num_image_tokens"] = config["text_config"]["num_image_tokens"]
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)


## Cleanup
import shutil
shutil.rmtree(TEMP_MODEL_OUTPUT_FOLDER)