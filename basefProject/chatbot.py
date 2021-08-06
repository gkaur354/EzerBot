#Chatbot application that uses the trained model

#Import libraries
import discord
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.models import load_model 
from nltk.tokenize import word_tokenize
import re
from string import punctuation

#Bot login 
client = discord.Client()
token = "ODcyOTE5MDc1OTM0ODMwNTkz.YQw3PQ.chMozVGpCcxhK8NHKGlLxhpCkWI"

#Load model and tokenizer
model = load_model('chatbotmodel.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Function that removes stopwords
with open("stopwords_pickle", "rb") as f:
    stopwords_list = pickle.load(f)
def remove_stopwords(phrase):
    noStopwords = ""
    tokenized_phrase = word_tokenize(phrase)
    for word in tokenized_phrase:
        if word not in stopwords_list:
            noStopwords += word
            noStopwords += " "
    return noStopwords

#Function that removes punctuation and numbers
def clean_text(phrase):
    pattern = r'[0-9]'
    final = ""
    for i in phrase:
        if i not in punctuation:
            new_word = re.sub(pattern, '', i)
            final += new_word
    return final 

#Function that classifies message
def classifyMessage(input):
    cleaned_phrase = clean_text(input.lower())
    noStopWords = remove_stopwords(cleaned_phrase)
    
    phrase = tokenizer.texts_to_sequences([noStopWords.lower()])
    padded = pad_sequences(phrase, maxlen=50, truncating='post', padding='post')
    pred = model.predict(padded)
    pred_int = pred.round().astype("int")
    if pred_int == 1:
        msg = "1"
    else:
        msg = "0"
    return msg

#Connect to Discord bot 
while True:
    @client.event
    async def on_ready():
        print("Ready")
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return 
        else:
            res = classifyMessage(message.content)
            if res != "":
                await message.reply(res)

            
    client.run(token)


