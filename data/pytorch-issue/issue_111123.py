causal_model = AutoModelForCausalLM.from_pretrained(model_pretrained_path_,
                                                    config=config,
                                                    trust_remote_code=True,
                                                    low_cpu_mem_usage=self.params["low_cpu_mem_usage"])

peft = PEFT(config_path_or_data=peft_params)
causal_model = peft.get_peft_model(model=causal_model)

trainer = Seq2SeqTrainer(
        params=trainer_params,
        model=causal_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=dataset_t.metric,
    )

trainer.train(resume_from_checkpoint=True)