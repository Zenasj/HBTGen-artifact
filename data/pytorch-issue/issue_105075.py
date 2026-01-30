import time
import timeit
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NUM_ITERS = 50

### FOLLOWING TUTORIAL ###

# load base model
model_id = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(model_id)

# create inputs
text = "This is just a sample text I am using as a test."
# inputs = tokenizer(text, padding="max_length", return_tensors="pt")
inputs = tokenizer(text, return_tensors="pt")

# compile the model
print("compiling the model...")
comp_model = torch.compile(base_model)
with torch.no_grad():
    comp_model(**inputs)

print("running timeit on eager mode...")
with torch.no_grad():
    # warmup
    for _ in tqdm(range(10)):
        base_model(**inputs)
    
    eager_t = timeit.timeit("base_model(**inputs)", number=NUM_ITERS, globals=globals())

print("running timeit on compiled mode...")
with torch.no_grad():
    # warmup
    for _ in tqdm(range(10)):
        comp_model(**inputs)
    
    inductor_t = timeit.timeit("comp_model(**inputs)", number=NUM_ITERS, globals=globals())


# now run on a set of unique texts
sample_size = 10
sample = random.sample(TEXTS, 10)

print("running inference on base model...")
with torch.no_grad():
    # warmup
    for _ in tqdm(range(10)):
        base_model(**inputs)
    
    base_total_time = 0
    for text in tqdm(sample):
        new_inputs = tokenizer(text, return_tensors='pt')
        start = time.time()
        base_model(**new_inputs)
        base_total_time += time.time() - start

print("running inference on compiled mode...")
with torch.no_grad():
    # warmup
    for _ in tqdm(range(10)):
        comp_model(**inputs)
    
    comp_total_time = 0
    for text in tqdm(sample):
        new_inputs = tokenizer(text, return_tensors='pt')
        start = time.time()
        comp_model(**new_inputs)
        comp_total_time += time.time() - start

print(f"eager repeat inputs: {eager_t * 1000 / NUM_ITERS} ms/iter")
print(f"inductor repeat inputs: {inductor_t * 1000 / NUM_ITERS} ms/iter")
print(f"speed up ratio: {eager_t / inductor_t}")
print(f"eager unique inputs: {(base_total_time / sample_size) * 1000} ms/iter")
print(f"inductor unique inputs: {(comp_total_time / sample_size) * 1000} ms/iter")
# compiling the model...
# running timeit on eager mode...
# 100%|██████████| 10/10 [00:01<00:00,  9.14it/s]
# running timeit on compiled mode...
# 100%|██████████| 10/10 [00:00<00:00, 36.74it/s]
# running inference on base model...
# 100%|██████████| 10/10 [00:00<00:00, 28.27it/s]
# 100%|██████████| 10/10 [00:00<00:00, 15.81it/s]
# running inference on compiled mode...
# 100%|██████████| 10/10 [00:00<00:00, 38.31it/s]
# 100%|██████████| 10/10 [02:17<00:00, 13.78s/it]
# eager repeat inputs: 35.97641162000059 ms/iter
# inductor repeat inputs: 25.08608974000026 ms/iter
# speed up ratio: 1.4341179511382955
# eager unique inputs: 61.79332733154297 ms/iter
# inductor unique inputs: 13781.427097320557 ms/iter