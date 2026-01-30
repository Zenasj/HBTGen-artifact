import torch

def exportable_broadcast(tensor1 : torch.Tensor, tensor2 : torch.Tensor):
    ''' broadcast tensors to the same shape using onnx exportable operators '''
    if len(tensor1.shape) < len(tensor2.shape):
        tensor2, tensor1 = exportable_broadcast(tensor2, tensor1)
    else:
        shape1 = tensor1.shape
        shape2 = tensor2.shape
        if len(shape1) == len(shape2):
            final_shape = [max(s1, s2) for s1, s2 in zip(shape1, shape2)]
            tensor1 = tensor1.expand(*final_shape)
            tensor2 = tensor2.expand(*final_shape)
        else:
            tensor2 = tensor2.expand(*shape1)
    return tensor1, tensor2

class ExportableNormal(Normal):
    def __init__(self, loc, scale, validate_args):
        self.loc, self.scale = exportable_broadcast(loc, scale)
        batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)