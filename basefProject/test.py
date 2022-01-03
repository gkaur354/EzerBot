#import libraries 
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import pickle

from keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model 
from nltk.tokenize import word_tokenize
import re
from string import punctuation

import csv
from typing import DefaultDict
import difflib
import requests


#Load test set 
data = pd.read_csv("/Users/gurnirmalkaur/Desktop/testSetBASEF.csv")
print(data.head())

model = load_model('chatbotmodel.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open("stopwords_pickle", "rb") as f:
    stopwords_list = pickle.load(f)

#Data preprocessing:
#Remove punctuation and numbers from text
def only_letters(text):
    import re
    pattern = r'[0-9]'
    from string import punctuation
    final = ""
    for i in text:
        if i not in punctuation:
            new_word = re.sub(pattern, '', i)
            final += new_word
    return final

data['phrase'] = data['phrase'].apply(lambda x:only_letters(x))

#Convert to lower case
data['phrase'] = data['phrase'].apply(lambda x: x.lower())

#Remove stopwords from text
def remove_stopwords(text):
    phrase = nltk.word_tokenize(text)
    noStopwords = ""
    for i in phrase:
        if i not in stopwords_list:
            noStopwords += i
            noStopwords += " "
    return noStopwords

data['phrase'] = data['phrase'].apply(lambda x:remove_stopwords(x))

X = data['phrase']
y = data['intention']

test_sequences = tokenizer.texts_to_sequences(X)
padded_test = pad_sequences(test_sequences, maxlen=50, truncating='post', padding='post')
