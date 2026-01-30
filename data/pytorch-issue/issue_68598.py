import torch.nn as nn

from transformers.onnx import OnnxConfig, PatchingSpec
from transformers.configuration_utils import PretrainedConfig
from typing import Any, List, Mapping, Optional, Tuple, Iterable
from collections import OrderedDict
from transformers import LayoutLMv2Processor
from datasets import load_dataset
from PIL import Image
import torch
from transformers import PreTrainedModel, TensorType
from torch.onnx import export
from transformers.file_utils import torch_version, is_torch_onnx_dict_inputs_support_available
from pathlib import Path
from transformers.utils import logging
from inspect import signature
from itertools import chain
from transformers import LayoutLMv2ForTokenClassification
from torch import nn
from torch.onnx import OperatorExportTypes

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LayoutLMv2OnnxConfig(OnnxConfig):
    def __init__(
            self,
            config: PretrainedConfig,
            task: str = "default",
            patching_specs: List[PatchingSpec] = None,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.max_2d_positions = config.max_2d_position_embeddings - 1

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("bbox", {0: "batch", 1: "sequence"}),
                ("image", {0: "batch"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    def generate_dummy_inputs(
            self,
            processor: LayoutLMv2Processor,
            batch_size: int = -1,
            seq_length: int = -1,
            is_pair: bool = False,
            framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:

        datasets = load_dataset("nielsr/funsd")
        example = datasets["test"][0]
        image = Image.open(example['image_path'])
        image = image.convert("RGB")

        if not framework == TensorType.PYTORCH:
            raise NotImplementedError("Exporting LayoutLM to ONNX is currently only supported for PyTorch.")

        input_dict = processor(image, example['words'], boxes=example['bboxes'], word_labels=example['ner_tags'],
                               return_tensors=framework)

        axis = 0
        for key_i in input_dict.data.keys():
            input_dict.data[key_i] = torch.cat((input_dict.data[key_i], input_dict.data[key_i]), axis)

        return input_dict.data


class pool_layer(nn.Module):
    def __init__(self):
        super(pool_layer, self).__init__()
        self.fc = nn.AvgPool2d(kernel_size=[8, 8], stride=[8, 8])

    def forward(self, x):
        output = self.fc(x)
        return output


def ensure_model_and_config_inputs_match(
        model: PreTrainedModel, model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """

    :param model:
    :param model_inputs:
    :return:
    """
    forward_parameters = signature(model.forward).parameters
    model_inputs_set = set(model_inputs)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # Make sure the input order match (VERY IMPORTANT !!!!)
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs


def export_model(
        processor: LayoutLMv2Processor, model: PreTrainedModel, config: LayoutLMv2OnnxConfig, opset: int, output: Path
) -> Tuple[List[str], List[str]]:
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        processor:
        model:
        config:
        opset:
        output:

    Returns:

    """

    if not is_torch_onnx_dict_inputs_support_available():
        raise AssertionError(f"Unsupported PyTorch version, minimum required is 1.8.0, got: {torch_version}")

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    with torch.no_grad():
        model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                logger.info(f"\t- {override_config_key} -> {override_config_value}")
                setattr(model.config, override_config_key, override_config_value)

        model_inputs = config.generate_dummy_inputs(processor, framework=TensorType.PYTORCH)
        inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
        print(matched_inputs)
        onnx_outputs = list(config.outputs.keys())

        if not inputs_match:
            raise ValueError("Model and config inputs doesn't match")

        config.patch_ops()
        model_inputs.pop("labels")
        export(
            model,
            (model_inputs,),
            f=output.as_posix(),
            input_names=list(config.inputs.keys()),
            output_names=onnx_outputs,
            dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
            do_constant_folding=True,
            use_external_data_format=config.use_external_data_format(model.num_parameters()),
            enable_onnx_checker=True,
            opset_version=opset,
            # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
        )

        config.restore_ops()

    return matched_inputs, onnx_outputs


if __name__ == '__main__':
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
    model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", torchscript = True)
    model.layoutlmv2.visual.pool = torch.nn.Sequential(pool_layer())
    onnx_config = LayoutLMv2OnnxConfig(model.config)
    export_model(processor=processor, model=model, config=onnx_config, opset=12, output=Path('onnx/layout.onnx'))

model_dir = "microsoft/layoutlmv2-base-uncased"
model = AutoModelForTokenClassification.from_pretrained(model_dir)
processor = LayoutLMv2Processor.from_pretrained(model_dir)
image = Image.open('./image.jpg').convert("RGB")
encoded_input = processor(
    image, return_tensors="pt"
)

traced_model = torch.jit.trace(func=model,
                                   strict=False,  
                                   example_inputs=[encoded_input['input_ids'], encoded_input['bbox'],
                                                   encoded_input['image'], encoded_input['attention_mask'],
                                                   encoded_input['token_type_ids']])