import os
from datasets import load_dataset, load_metric, Features, Value
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load your dataset
output_file = "qa_dataset.json"

data_files = {"train": output_file}
# defining the features parameter based on the schema/structure of the QA dataset in json format
features = Features({
    "question": Value(dtype="string"),
    "answer": Value(dtype="string")
})

print(features)

dataset = load_dataset("json", data_files=data_files, features=features)

# Loading the pre-trained tokenizer and model
model_name = "bert-base-uncased"  # You can change this to another model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenize the inputs
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_index = sample_map[i]
        answer = answers[sample_index]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_pos = [o[0] for o in offset].index(start_char)
            end_pos = [o[1] for o in offset].index(end_char)
            start_positions.append(start_pos)
            end_positions.append(end_pos)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)