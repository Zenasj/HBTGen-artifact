from transformers import AutoModel, AutoTokenizer
from optimum import bettertransformer
import torch
import optimum.version
import transformers

print(torch.__version__, transformers.__version__, optimum.version.__version__)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


text = "Hello this is a test"
inputs = tokenizer(text, return_tensors="pt").to("cuda") 
# something is wrong here, I forgot how to reproduce above code.
inputs.pop("token_type_ids")

for bettertransform in [False, True]:
    for compile in [True, False]:
        with torch.no_grad():
            bert = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to("cuda").to(torch.bfloat16)
            if bettertransform:
                bert = bettertransformer.BetterTransformer.transform(bert)

            if compile:
                bert = torch.compile(bert)
            else:
                print("no compile")
            outputs = bert(**inputs)
            print(bettertransform, compile,
                outputs.last_hidden_state.shape, outputs.last_hidden_state.mean()
            )