def forward(self, input_to_slice, input_to_copy):
    [...]
    return input_to_slice, input_to_copy

def forward(self, *input_to_slice, **kwargs_to_copy):
    [...]
    return input_to_slice, kwargs_to_copy

def forward_stage1(tensor, foo, bar, baz):
  pass

Pipe.forward(tensor, foo=1, bar=2, baz=2)

aggregate = ()
for block in layers: # set of 6 identical blocks (t5stack)
    output = block(input)
    aggregate += output[1]

output = [0, 1]
aggregate = 2

output = [2, 3]
aggregate = 6

import sys
# edit this to point to where you checked out transformers if it's not under /tmp/transformers - "src" is where the source is
sys.path.insert(0, "/tmp/transformers/src")

from transformers import T5Tokenizer, T5ForConditionalGeneration
mname = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(mname)
model = T5ForConditionalGeneration.from_pretrained(mname, return_dict=True)

texts = ["This is good", "This is bad"]
texts = ["translate English to French: "+x for x in texts]
batch = tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt")
outputs = model.generate(**batch)
for x in outputs:
    decoded = tokenizer.decode(x, skip_special_tokens=True)
    print(decoded)