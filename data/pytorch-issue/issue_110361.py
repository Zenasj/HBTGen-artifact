import torchvision

from collections import OrderedDict
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sd_key = 'features.0.1.num_batches_tracked'
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1.DEFAULT)
print(f"before loading: {model.state_dict()['features.0.1.num_batches_tracked']}")

empty_dict = OrderedDict()
model.load_state_dict(empty_dict, strict=False)
print(f"after loading: {model.state_dict()['features.0.1.num_batches_tracked']}")