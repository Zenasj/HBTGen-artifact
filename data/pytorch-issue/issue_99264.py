from transformers import AutoModel


model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b-int8", trust_remote_code=True, device_map='auto', revision='9076e37'
)