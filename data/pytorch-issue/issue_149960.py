tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained_model, None, model_name, device_map=device_map, attn_implementation="eager",
    )