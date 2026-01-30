def fsdp_main(args=None):
    model_name = "facebook/opt-350m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OPTForSequenceClassification.from_pretrained(model_name)

if __name__ == '__main__':
    fsdp_main()