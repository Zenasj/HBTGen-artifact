def delete_end_dots(input_file_name, output_file):
    with open(input_file_name, "r", encoding="utf-8") as input_f, open(output_file, "w+", encoding="utf-8") as out_f:
        for line in input_f:
            line = line.strip()
            splitted = line.split(" ")
            if splitted[-1] == '.':
                splitted = splitted[:-1]
            line = " ".join(splitted)
            print(line, file=out_f)


delete_end_dots("sample-corpus-asl-en.asl.txt", "asl_processed.txt")
delete_end_dots("sample-corpus-asl-en.en.txt", "english_processed.txt")

"""aslg_dataset dataset."""

import sys
print("started on kernel {}".format(print(sys.executable)))

import tensorflow_datasets as tfds
import re
import tensorflow.io.gfile as gfile

# TODO(aslg_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
## ASLG-SMT Dataset 
## by Achraf Othman and Mohamed Jemni
This dataset has 87706 sentences, in both english and American Sign Language, glossed.
We added some processing compared to the original dataset: punctuation is split from the words for english (some sentences did not apply this rule).
we also assert that each word is separated from anything else, for both (e.g. 123word becomes 123 word).
We finally delete any duplicate spaces. 
"""

# TODO(aslg_dataset): BibTeX citation
_CITATION = """@INPROCEEDINGS{8336054,
author={A. {Othman} and M. {Jemni}},
booktitle={2017 6th International Conference on Information and Communication Technology and Accessibility (ICTA)},
title={An XML-gloss annotation system for sign language processing},
year={2017},
volume={},
number={},
pages={1-7},
doi={10.1109/ICTA.2017.8336054}}
"""


class AslgDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for aslg_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(aslg_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'input_text': tfds.features.Text(),
            'output_text': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        citation=_CITATION,
    )

  def _split_generators(self, _):
    """Returns SplitGenerators."""
    # TODO(aslg_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(aslg_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        # 'train': self._generate_examples(path / 'train_imgs'),
        'train': self._generate_examples()
    }

  def _generate_examples(self):
    """Yields examples.
    File encoding is utf-8-sig"""
    english = gfile.GFile('english_processed_utf8.txt', 'rb').read() ## I tried to open the file, and then use readlines, but it produces the same error
    asl = gfile.GFile('asl_processed_utf8.txt', 'rb').read() ## '_utf8' in the file name describes the same file as without, except it was saved with an utf-8 encoding instead of utf-8-sig. The exact same behaviour occurs with both files (with and without '_utf8')
    # with open('english_processed.txt', encoding='utf-8-sig') as english_f, open('asl_processed.txt', encoding='utf-8-sig') as asl_f:
    english = [eng for eng in english.split('\n') if eng]
    asl = [a for a in asl.split('\n') if a]
    for i, sentences in enumerate(zip(english, asl)):
      eng_sen, asl_sen = self._clean_sentences(*sentences)
      yield i, {
        'input_text': eng_sen.strip(),
        'output_text': asl_sen.strip()
      }

  @staticmethod
  def _clean_sentences(eng_sen, asl_sen):
    eng = eng_sen.replace("!", " ! ").replace(".", " . ").replace(",", " , ")
    eng = re.sub(r'([\-\'a-zA-ZÀ-ÖØ-öø-ÿ]+)', r' \1 ', eng)
    eng = re.sub(r' +', ' ', eng)

    asl = re.sub(r'([0-9]+(?:[.][0-9]*)?)', r' \1 ', asl_sen)
    asl = re.sub(r' +', ' ', asl)

    return eng, asl