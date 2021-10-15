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

import csv
import collections
from types import CoroutineType
from typing import DefaultDict
import difflib
import requests

#List of cities in Canada 
cities = DefaultDict(list)
with open('/Users/gurnirmalkaur/Desktop/canadaCities.csv', "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        city = row[1]
        province = row[0]
        cities[city].append(province)

#Bot login 
client = discord.Client()
token = "ODcyOTE5MDc1OTM0ODMwNTkz.YQw3PQ.chMozVGpCcxhK8NHKGlLxhpCkWI"


model = load_model('chatbotmodel.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
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

    return pred_int


#Function that corrects spelling of city 
def correct_spelling(user_response):
    city_names = list(cities.keys())
    response = user_response.capitalize()
    if city_names.count(response):
        return response 
    else:
        matches = difflib.get_close_matches(response, city_names)
        return matches[0]

#Function that finds resources in the user's vicinity using Yelp's API

url = 'https://api.yelp.com/v3/businesses/search'
key = open(r'/Users/gurnirmalkaur/Desktop/key.txt').readlines()[0]

def find_resource(city):
    location = correct_spelling(str(city))
    category = "c_and_mh"
    term = "mental health"
    resources = ""
    headers = {
    'Authorization': 'Bearer %s' % key
    }       
    parameters = {'location': location,
                'categories': category,
                'term': term,
                'limit': 5}
    response = requests.get(url, headers=headers, params=parameters)
    resource = response.json()['businesses']

    resources = "Here are some mental health resources in your area that you may find helpful:\n"
    for r in resource:
        name = r['name']
        location = r['location']['address1']
        city = r['location']['city']
        province = r['location']['state']
        resource = (f'{name}, {location}, {city}, {province}\n')
        resources += resource 
    return resources

def check_answer1(msg):
    if msg.lower() == "y":
        reply = "add resources"
    elif msg.lower() == "n":
        reply = "ask about anxiety/depression (Y/N)"
    else:
        reply = "Sorry, I don't understand"
    return reply

def check_answer2(response):
    if response.lower() == "y":
        reply = "ask city"
    elif response.lower() == "n":
        reply = "somethign"
    else:
        reply = "Sorry, I don't understand"
    return reply


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
            user = message.author
            res = classifyMessage(message.content)
            if res[0][0] == 1:
                await message.reply("Hi! My name is Hera. I noticed you may be experiencing emotional distress. If so, I want to help you. Are you having suicidal thoughts? (Y/N)")    
                msg = await client.wait_for("message", check=lambda m: m.author == user)
                await msg.reply(check_answer1(msg.content))
                response = await client.wait_for("message", check=lambda m: m.author == user)
                await response.reply(check_answer2(response.content))
                city = await client.wait_for("message", check=lambda m: m.author == user)
                await city.reply(find_resource(city.content))
                       
    client.run(token)



