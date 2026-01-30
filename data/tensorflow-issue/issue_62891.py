py
from abc import ABC

import tensorflow as tf
from datasets import load_dataset
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

lang_dict = {
    tf.constant('en').ref(): 50259,
    tf.constant('zh').ref(): 50260,
    tf.constant('de').ref(): 50261,
    tf.constant('es').ref(): 50262,
    tf.constant('ru').ref(): 50263,
    tf.constant('ko').ref(): 50264,
    tf.constant('fr').ref(): 50265,
    tf.constant('ja').ref(): 50266,
    tf.constant('pt').ref(): 50267,
    tf.constant('tr').ref(): 50268,
    tf.constant('ar').ref(): 50272,
    tf.constant('it').ref(): 50274,
    tf.constant('ur').ref(): 50290,
    tf.constant('fa').ref(): 50300,
    tf.constant('th').ref(): 50289,
    tf.constant('id').ref(): 50275
}


def save_tf_model(model):
    processor = WhisperProcessor.from_pretrained("./whisper-base")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("./whisper-base")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("./whisper-base", predict_timestamps=True)
    processor = WhisperProcessor(feature_extractor, tokenizer)
    # Loading dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = processor(
        ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="tf"
    )
    input_features = inputs.input_features

    # Generating Transcription
    generated_ids = model.generate(input_features=input_features, forced_decoder_ids=forced_decoder_ids)
    print(generated_ids)
    transcription = processor.tokenizer.decode(generated_ids[0])
    print(transcription)
    model.save('./content/tf_whisper_saved')


def convert_tflite(model):
    saved_model_dir = './content/tf_whisper_saved'
    tflite_model_path = './whisper-base.tflite'

    generate_model = GenerateModel(model=model)
    tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


class GenerateModel(tf.Module):
    def __init__(self, model):
        super(GenerateModel, self).__init__()
        self.model = model
        self.lang_dict = {
            tf.constant(50259).ref(): 50259,
            tf.constant(50260).ref(): 50260,
            tf.constant(50261).ref(): 50261,
            tf.constant(50262).ref(): 50262,
            tf.constant(50263).ref(): 50263,
            tf.constant(50264).ref(): 50264,
            tf.constant(50265).ref(): 50265,
            tf.constant(50266).ref(): 50266,
            tf.constant(50267).ref(): 50267,
            tf.constant(50268).ref(): 50268,
            tf.constant(50272).ref(): 50272,
            tf.constant(50274).ref(): 50274,
            tf.constant(50290).ref(): 50290,
            tf.constant(50300).ref(): 50300,
            tf.constant(50289).ref(): 50289,
            tf.constant(50275).ref(): 50275
        }

    @tf.function(
        # shouldn't need static batch size, but throws exception without it (needs to be fixed)
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
            tf.TensorSpec((), tf.int32, name="lang")
        ],
    )
    def serving(self, input_features, lang):
        outputs = self.model.generate(
            input_features,
            max_new_tokens=450,  # change as needed
            return_dict_in_generate=True,
            forced_decoder_ids=[(1, self.lang_dict.get(lang.ref(), 50259)), (2, 50359), (3, 50363)]
        )
        return {"sequences": outputs["sequences"]}


def test():
    tflite_model_path = 'whisper-base.tflite'
    feature_extractor = WhisperFeatureExtractor.from_pretrained("./whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("./whisper-base", predict_timestamps=True)
    processor = WhisperProcessor(feature_extractor, tokenizer)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = processor(
        ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="tf"
    )
    input_features = inputs.input_features

    interpreter = tf.lite.Interpreter(tflite_model_path)

    tflite_generate = interpreter.get_signature_runner()
    generated_ids = tflite_generate(input_features=input_features, lang=tf.constant(50259))["sequences"]
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)


if __name__ == '__main__':
    model = TFWhisperForConditionalGeneration.from_pretrained("./whisper-base")
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    # print(forced_decoder_ids)

    save_tf_model(model)
    convert_tflite(model)
    test()