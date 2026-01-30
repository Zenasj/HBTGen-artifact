import torch
from transformers import BertForSequenceClassification
from accelerate import Accelerator

checkpoint_path = "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_original-c1212f89.ckpt"
model_type = "bert-base-uncased"
num_classes = 6

accelerator = Accelerator()
loaded = torch.hub.load_state_dict_from_url(
    checkpoint_path, map_location=accelerator.device
)
state_dict = loaded["state_dict"]
config = BertForSequenceClassification.config_class.from_pretrained(
    model_type, num_labels=num_classes
)
model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=state_dict,
    local_files_only=False,
)
print(type(model))