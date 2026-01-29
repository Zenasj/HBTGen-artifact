# tf.random.uniform((1, 80, 3000), dtype=tf.float32) ← inferred input shape of input_features 

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model
        
        # lang_dict maps lang tensor refs to token IDs for forced decoder ids
        # The dictionary keys and values are token IDs (int constants)
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
        input_signature=[
            tf.TensorSpec(shape=(1, 80, 3000), dtype=tf.float32, name="input_features"),
            tf.TensorSpec(shape=(), dtype=tf.int32, name="lang")
        ]
    )
    def call(self, input_features, lang):
        # Generate sequences with forced_decoder_ids to control language and task
        forced_ids = [
            (1, self.lang_dict.get(lang.ref(), 50259)),  # language token forced at position 1
            (2, 50359),                                  # special task token for transcribe?
            (3, 50363)                                   # special task token for timestamps?
        ]
        outputs = self.model.generate(
            input_features=input_features,
            max_new_tokens=450,
            return_dict_in_generate=True,
            forced_decoder_ids=forced_ids
        )
        # Return the generated sequence IDs as output
        return outputs["sequences"]

def my_model_function():
    """
    Returns an instance of MyModel wrapping the TFWhisperForConditionalGeneration model.
    Assumes 'transformers' and whisper-base model files are available under ./whisper-base.
    """
    from transformers import TFWhisperForConditionalGeneration
    
    # Load the pretrained whisper model from local directory
    model = TFWhisperForConditionalGeneration.from_pretrained("./whisper-base")
    return MyModel(model)

def GetInput():
    """
    Returns a random input_features tensor with shape (1, 80, 3000) and a language token int32 scalar.
    These inputs are compatible with MyModel.
    
    Assumptions:
    - batch size = 1
    - input_features dims from the original code: (1, 80, 3000)
    - lang token int32, default 50259 ('en' English)
    """
    input_features = tf.random.uniform((1, 80, 3000), dtype=tf.float32)
    lang = tf.constant(50259, dtype=tf.int32)  # English language token
    return input_features, lang

