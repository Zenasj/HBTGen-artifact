import argparse
import time
import torch
import torch._dynamo
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = False

from transformers import AutoModelForSeq2SeqLM, pipeline, AutoTokenizer

# torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

model_id = "google/flan-t5-base"

kwargs = dict(torch_dtype=torch.bfloat16, use_cache=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()

input_sentence = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and"

generation_kwargs = dict(max_length=128, min_length=128, do_sample=False, num_beams=4, num_beam_groups=1, no_repeat_ngram_size=2)
print(input_sentence)
print(f"input tokens num is {len(tokenizer(input_sentence)['input_ids'])}")
inputs = tokenizer(input_sentence, return_tensors='pt')

with torch.no_grad():
    for i in range(8):
        pre = time.time()
        output = model.generate(**inputs, **generation_kwargs)
        print(f"eager eval time {i}: {time.time()-pre}")

print("Use torch compile model")
model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)
with torch.no_grad():
    for i in range(8):
        pre = time.time()
        output_compile = model.generate(**inputs, **generation_kwargs)
        print(f"compile eval time {i}: {time.time()-pre}")