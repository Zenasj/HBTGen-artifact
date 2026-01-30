import torch

inputs = {
    'feature_a': torch.Tensor([[101, 7770, 1107, 4638, 5381, 2605, 4511, 1351, 102, 0]]),
    'feature_b': torch.Tensor([[101, 7770, 1107, 4638, 5381, 2605, 4511, 1351, 102, 0]])
}

def fn_simple_dict(inputs):
    decoder_outputs = {'feature_a': inputs['feature_a'], 'feature_b': inputs['feature_b']}
    predictions = {
        'feature_a': inputs['feature_a'],
        'feature_b': inputs['feature_b']
    }
    return predictions, decoder_outputs

torch.jit.trace(fn_simple_dict, inputs, strict=False)

import torch

inputs = {
    'feature_a': torch.Tensor([[101, 7770, 1107, 4638, 5381, 2605, 4511, 1351, 102, 0]]),
    'feature_b': torch.Tensor([[101, 7770, 1107, 4638, 5381, 2605, 4511, 1351, 102, 0]])
}


def fn_nested_dict(inputs):
    decoder_outputs = {'feature_a': inputs['feature_a'], 'feature_b': inputs['feature_b']}
    predictions = {
        'feature_a': {'targets': inputs['feature_a'], 'probs': inputs['feature_a']},
        'feature_b': {'targets': inputs['feature_b'], 'probs': inputs['feature_b']}
    }
    return predictions, decoder_outputs

torch.jit.trace(fn_nested_dict, inputs, strict=False)