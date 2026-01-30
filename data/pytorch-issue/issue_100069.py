import torch.nn as nn

import torch
import transformers

from composer.utils import dist

def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True
    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False

def main():
    # initialize dist
    dist.initialize_dist(None)

    # load base model and tokenizer from Hugging Face
    gpt = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m')
    gptt = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')

    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    # This seems to cause other problems...
    # for module in gpt.modules():
    #     module._fsdp_wrap = True

    gpt._fsdp_wrap = True
    
    # move model to gpu
    gpt.to(torch.cuda.current_device())
    # FSDP wrap
    fsdp_wrapped_gpt = FSDP(gpt, auto_wrap_policy=_auto_wrap_policy, use_orig_params=False)
    print(fsdp_wrapped_gpt)

    # create the input
    input_dict = gptt('hello', return_tensors='pt')
    input_dict['input_ids'] = input_dict['input_ids'].to(torch.cuda.current_device())
    input_dict['attention_mask'] = input_dict['attention_mask'].to(torch.cuda.current_device())

    # THIS CODE IS NECESSARY IN ORDER FOR .generate TO NOT ERROR BELOW (THIS WAS A PREVIOUS WORKAROUND FROM TORCH 1.13 THAT STILL SEEMS TO BE NECESSARY)
    with torch.no_grad():
        fsdp_wrapped_gpt.forward(input_ids=input_dict['input_ids'])

    # call generate
    generation = fsdp_wrapped_gpt.generate(input_ids=input_dict['input_ids'], attention_mask=input_dict['attention_mask'], max_new_tokens=5)
    print(generation)

if __name__ == '__main__':
    main()