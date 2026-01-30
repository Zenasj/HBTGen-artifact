from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import torch

def convert_gpt2_model_to_onnx() -> str:
    """
    convert pytorch model to onnx format
    """
    # Load model and set the model to eval mode
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('sshleifer/tiny-gpt2')
    model.eval()

    # batch_size, input_ids_length and past_sequence_length are dynamic axes
    # We have to initialize a random input(the value doesn't matter) for the model, because the converting requires execution of the model
    # See: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    batch_size = 3
    input_ids_length = 2
    past_sequence_length = 1
    config: GPT2Config = model.config
    num_attention_heads = config.n_head
    hidden_size = config.n_embd
    num_layer = config.n_layer
    vocab_size = config.vocab_size
    config.is_decoder = True

    # past(`past_key_values` in model) is a list, its length is num_layer.
    # each element in the list is a tuple(key, value), and key/value's shape is past_shape,
    # aka [batch_size, n_heads, past_sequence_length, embd_size_each_head]
    past_shape = [batch_size, num_attention_heads,
                  past_sequence_length, int(hidden_size/num_attention_heads)]
    past = [(torch.rand(past_shape, dtype=torch.float32, device='cpu'), torch.rand(past_shape, dtype=torch.float32, device='cpu'))
            for _ in range(num_layer)]

    # input_ids is a [batch_length, input_ids_length] tensor
    input_ids = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, input_ids_length),
        dtype=torch.long,
        device='cpu',
    )

    # attention_mask is a 0/1 tensor of [batch_size, past_sequence_length + input_ids_length]
    attention_mask = torch.ones(
        [batch_size, past_sequence_length + input_ids_length]).to(torch.long)

    # token_type_ids is not needed in our case, its size is [batch_size, input_ids_length]
    token_type_ids = torch.zeros([batch_size, input_ids_length]).to(torch.long)

    # position_ids, size is [batch_size, input_ids_length]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids[:, past_sequence_length:].to(torch.long)

    # Run the model and get output from the model
    output = model(input_ids, past_key_values=past, attention_mask=attention_mask,
                   position_ids=position_ids, return_dict=True, use_cache=True)

    # Set output names
    output_names = ['logits']
    for i in range(num_layer):
        output_names.append("present_key" + str(i))
        output_names.append("present_value" + str(i))

    # Set input_names
    input_names = ['input_ids']
    for i in range(num_layer):
        input_names.append("past_key" + str(i))
        input_names.append("past_value" + str(i))
    input_names += ['attention_mask', 'position_ids']

    # Set dynamic axes
    dynamic_axes = {}
    dynamic_axes['input_ids'] = {0: 'batch_size', 1: 'input_ids_length'}
    dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'total_length'}
    dynamic_axes['position_ids'] = {0: 'batch_size', 1: 'input_ids_length'}
    dynamic_axes['logits'] = {0: 'batch_size', 1: 'input_ids_length'}
    for i in range(num_layer):
        dynamic_axes['past_key' +
                     str(i)] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes['past_value' +
                     str(i)] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes['present_key' +
                     str(i)] = {0: 'batch_size', 2: 'total_length'}
        dynamic_axes['present_value' +
                     str(i)] = {0: 'batch_size', 2: 'total_length'}

    # The first input is required, and other inputs can be passed to torch.onnx.export using dict
    inputs = (input_ids, {
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'past_key_values': past,
    })

    # Do export using torch.onnx.export
    exported_model = "converted_model.onnx"
    torch.onnx.export(
        model,
        args=inputs,
        f=exported_model,
        export_params=True,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=15,
        export_modules_as_functions=True
    )
    return exported_model

convert_gpt2_model_to_onnx()