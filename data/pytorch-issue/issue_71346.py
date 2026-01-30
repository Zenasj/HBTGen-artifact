# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

import random, pickle, os, csv
import re, string
import string
#import stanza
import spacy_stanza
import warnings
warnings.filterwarnings("error")
from random import shuffle

# stanza.download('fr')
nlp = spacy_stanza.load_pipeline('fr', processors='tokenize,mwt,pos,lemma')
random.seed(1)

def tokenizer(sentence):

	sent_doc = nlp(sentence)
	wds = [token.text for token in sent_doc if token.pos_ != 'SPACE']
	return wds