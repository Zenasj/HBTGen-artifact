from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

text = 'Był to świetny pomysł, bo punktował Prawo i Sprawiedliwość tam, gdzie jest ono najsłabsze, mimo że udaje najsilniejsze. Uderzał w wizerunek państwa dobrobytu, które nikogo nie zostawia z tyłu i wyrównuje szanse. Tutaj mamy pewnego rodzaju déjà vu.'

vectorize_layer = TextVectorization()
vectorize_layer.adapt([text])
print(vectorize_layer.get_vocabulary())

values = [1]
keys = [b'warszawie\xc2']
[x.decode('utf-8') for _, x in sorted(zip(values, keys))]

def _get_vocabulary():
    keys, values = vectorize_layer._index_lookup_layer._table_handler.data()
    return [x.decode('utf-8', errors='ignore') for _, x in sorted(zip(values, keys))]

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)

text = 'Był to świetny pomysł, bo punktował Prawo i Sprawiedliwość tam, gdzie jest ono najsłabsze, mimo że udaje najsilniejsze. Uderzał w wizerunek państwa dobrobytu, które nikogo nie zostawia z tyłu i wyrównuje szanse. Tutaj mamy pewnego rodzaju déjà vu.'

vectorize_layer = TextVectorization()
vectorize_layer.adapt([text])
print(vectorize_layer.get_vocabulary())

class OutputTextProcessor(TextVectorization):
    def __init__(self):
        super(OutputTextProcessor, self).__init__(standardize=self.tf_lower_and_split_punct)

    def tf_lower_and_split_punct(self, text): # name of the function is from example on https://www.tensorflow.org/text/tutorials/nmt_with_attention#the_encoderdecoder_model
        # Strip whitespace.
        text = tf.strings.strip(text)

        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
        return text
    def get_vocabulary(self):
        # _layer.get_vocabulary
        keys, values = self._lookup_layer.lookup_table.export()
        # print(self._lookup_layer.lookup_table.export())
        vocab = []
        for i in keys : 
            try :
                vocab.append(i.numpy().decode('utf-8'))
            except :
                vocab.append(i.numpy().decode('ISO-8859-1'))
        return vocab