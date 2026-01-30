import time

from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2-large')
set_seed(42)

start_time = time.time()

generator("Hello, I'm a language model", max_length=40, num_return_sequences=1)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")