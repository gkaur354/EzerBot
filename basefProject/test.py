#import libraries 
import nltk
import pandas as pd
import pickle

from keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#Load test set 
data = pd.read_csv("/Users/gurnirmalkaur/Desktop/pathBASEF/testingsetBASEF.csv")

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

 

# predict on test set
classes = model.predict_classes(padded_test, verbose=0)
print(classes)
# reduce to 1d array

yhat_classes = classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y, classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y, classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y, classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y, classes)
print('F1 score: %f' % f1)

# confusion matrix
matrix = confusion_matrix(y, classes)
print(matrix)

