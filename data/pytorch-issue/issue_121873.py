import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

tokenizer.pad_token = "!" #Not EOS, will explain another time.\

CUTOFF_LEN = 256  #Our dataset has shot text

dataset = load_dataset("harpreetsahota/modern-to-shakesperean-translation") #Found a good small dataset for a quick test run! Thanks to the uploader!
train_data = dataset["train"] # Not using evaluation data

def generate_prompt(user_query):
    sys_msg= "Translate the given text to Shakespearean style."
    p =  "<s> [INST]" + sys_msg +"\n"+ user_query["modern"] + "[/INST]" +  user_query["shakespearean"] + "</s>"
    return p 

def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN ,
        padding="max_length"
    )

train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)), remove_columns=["modern" , "shakespearean"])
opt_model = torch.compile(model)
trainer = Trainer(
    model=opt_model,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=100,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_torch",
        save_strategy="epoch",
        output_dir="mixtral-moe-lora-instruct-shapeskeare",
        remove_unused_columns=False
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

trainer.train()