import torch
import torch._dynamo as torch_dynamo
import torchvision.models as models
import torch.ao.quantization.pt2e.duplicate_dq_pass

from torch._inductor import config

config.freezing = True

@torch_dynamo.register_backend(name="tmp")
def tmp_compile(gm, inputs, **kwargs):

    # print("before const fold: \n", gm.graph)
    # return gm

    import torch._inductor.constant_folding
    torch._inductor.constant_folding.constant_fold(gm)

    # print("after const fold: \n", gm.graph)

    return gm

def load_model():
    model = models.swin_v2_s(weights=True)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model()

    x = torch.randn(1, 3, 224, 224)

    x_cp = x.detach().clone()
    import copy
    model_cp = copy.deepcopy(model)
    eager_out = model_cp(x_cp)

    sm = torch.compile(model, backend="tmp")

    with torch.inference_mode():
        out = sm(x)

    torch.testing.assert_close(eager_out, out, atol=1e-5, rtol=1e-5)