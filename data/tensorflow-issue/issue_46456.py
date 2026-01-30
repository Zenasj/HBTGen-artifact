import tensorflow as tf

class MyLookupModel(tf.train.Checkpoint):
  def __init__(self, vocab_file):
    super().__init__()
    vocab_initializer = tf.lookup.TextFileInitializer(
        vocab_file,
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    self._vocab_table = tf.lookup.StaticHashTable(vocab_initializer,
                                                  default_value=-1)

  @tf.function(input_signature=[tf.TensorSpec((None,), tf.string)])
  def __call__(self, inputs):
    return self._vocab_table.lookup(inputs)

ORIGINAL_VOCAB = "/tmp/original/vocab.txt"
tf.io.gfile.makedirs(os.path.dirname(ORIGINAL_VOCAB))
with tf.io.gfile.GFile(ORIGINAL_VOCAB, "w") as f:
  for x in ["a", "b", "c", "d"]:
    f.write(x + "\n")

model0 = MyLookupModel(ORIGINAL_VOCAB)
tf.saved_model.save(model0, "/tmp/model1")
model1 = tf.saved_model.load("/tmp/model1")
tf.saved_model.save(model1, "/tmp/model2")
# If "/tmp/model1/assets/vocab.txt" is deleted at this point, the next line crashes.
model2 = tf.saved_model.load("/tmp/model2")