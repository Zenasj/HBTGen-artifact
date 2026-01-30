import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from srcnew.utils.tokenizerT5 import get_default_tokenizer, tokenizer_T5_special_tokens
import torch
import os

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 模型和分词器路径
model_path = "/home/liuwenlong/KKCMG/RACE/saved_model/codet5/chronicle/checkpoint-best-bleu/pytorch_model.bin"


encoder, tokenizer = None, None

def load_encoder(model_name="Salesforce/codet5-base", model_path=None, special_tokens=None):
    # 加载分词器和模型
    tokenizer = get_default_tokenizer(model_name=model_name, special_tokens=special_tokens)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    # 加载模型权重
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    
    return model.get_encoder(), tokenizer

def get_hidden_representation(encoder, tokenizer, input_text):
    # 准备输入
    # inputs = tokenizer(input_text, return_tensors="pt").to(device)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # 获取编码器输出
    with torch.no_grad():
        encoder_outputs = encoder(**inputs)
        text_vector = encoder_outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    
    # 取最后一个序列的 hidden_size
    last_hidden = text_vector[0][0]
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    torch.cuda.empty_cache()
    return text_vector, last_hidden

def get_batch_hidden_representation(encoder, tokenizer, input_text_list):
    # 准备输入
    inputs = tokenizer(input_text_list, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 获取编码器输出
    with torch.no_grad():
        encoder_outputs = encoder(**inputs)
        text_vector = encoder_outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    
    # 取最后一个序列的 hidden_size
    last_hidden = text_vector[:, 0, :]
    return text_vector, last_hidden