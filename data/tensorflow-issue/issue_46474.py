# tf.random.uniform((B, ) ...) ← This dataset code example uses text file input, so no direct tensor input shape is defined.

import tensorflow as tf
import tensorflow.io.gfile as gfile
import tensorflow_datasets as tfds
import re


class MyModel(tf.keras.Model):
    """
    Adapted from the ASLG-SMT Dataset Builder logic.
    This model serves as a placeholder demonstrating:
    - Reading UTF-8 text files via GFile with 'latin-1' fallback decoding
    - Basic preprocessing of English and American Sign Language glossed sentences.
    
    Since the original issue and code revolve around reading and processing text files for dataset creation,
    this MyModel reads and cleans the text lines from two expected files, and yields paired processed strings.
    
    Note:
        This is designed as a tf.keras.Model for the sake of the task structure,
        but in practice this kind of data ingestion and preprocessing is better suited 
        in a tf.data pipeline or a custom tfds dataset builder class.
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """
        Accepts no inputs, returns a tf.Tensor of string pairs (english, asl).
        
        To adhere to TF model conventions and enable usage with tf.function,
        inputs argument expected but ignored.
        """

        # We will read files, decode with 'utf-8' first, fallback to 'latin-1' for problematic chars.
        # Returns a tensor of shape (N, 2) where each row is [english_sentence, asl_sentence]

        def read_file_lines(filename):
            try:
                # Try reading as utf-8
                with gfile.GFile(filename, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
            except UnicodeDecodeError:
                # Fallback: read raw bytes, decode with latin-1 which won't error but may map bytes differently
                with gfile.GFile(filename, 'rb') as f:
                    raw_bytes = f.read()
                lines = raw_bytes.decode('latin-1').splitlines()
            return [line.strip() for line in lines if line.strip()]

        # Read and preprocess English and ASL sentences from files
        english_lines = read_file_lines('english_processed_utf8.txt')
        asl_lines = read_file_lines('asl_processed_utf8.txt')

        # To handle any length mismatch gracefully, zip to shortest length
        n = min(len(english_lines), len(asl_lines))

        eng_cleaned = []
        asl_cleaned = []

        for i in range(n):
            eng, asl = self._clean_sentences(english_lines[i], asl_lines[i])
            eng_cleaned.append(eng)
            asl_cleaned.append(asl)

        # Convert to tf.Tensor of shape (n, 2) with dtype=tf.string
        paired = tf.stack([tf.constant(eng_cleaned), tf.constant(asl_cleaned)], axis=1)
        return paired

    @staticmethod
    def _clean_sentences(eng_sen, asl_sen):
        # For English: split punctuation, separate words, collapse spaces
        eng = eng_sen.replace("!", " ! ").replace(".", " . ").replace(",", " , ")
        eng = re.sub(r'([\-\'a-zA-ZÀ-ÖØ-öø-ÿ]+)', r' \1 ', eng)
        eng = re.sub(r' +', ' ', eng)

        # For ASL: separate numbers, collapse spaces
        asl = re.sub(r'([0-9]+(?:[.][0-9]*)?)', r' \1 ', asl_sen)
        asl = re.sub(r' +', ' ', asl)

        return eng.strip(), asl.strip()


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    """
    Since MyModel's call ignores input (reads from files directly),
    we return a dummy tensor to satisfy signature.
    """
    # Return a dummy float tensor of shape (1,) - arbitrary, not actually used by model in this case.
    return tf.random.uniform((1,), dtype=tf.float32)

