tokenizer = tr.XLMRobertaTokenizer.from_pretrained("xlm-roberta-large",local_files_only=True)
model = tr.XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large", return_dict=True,local_files_only=True)
model.gradient_checkpointing_enable() #included as new line

training_args = tr.TrainingArguments(

     output_dir='****'
    ,logging_dir='****'        # directory for storing logs
    ,save_strategy="epoch"
    ,run_name="****"
    ,learning_rate=2e-5
    ,logging_steps=1000
    ,overwrite_output_dir=True
    ,num_train_epochs=10
    ,per_device_train_batch_size=8
    ,prediction_loss_only=True
    ,gradient_accumulation_steps=4
#     ,gradient_checkpointing=True
    ,bf16=True #57100 
,optim="adafactor"

)


trainer = tr.Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data
)