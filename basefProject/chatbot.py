#Chatbot application that uses the trained model

#Import libraries
import discord
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

#List of cities in Canada 
cities = DefaultDict(list)
with open('/Users/gurnirmalkaur/Desktop/pathBASEF/canadaCities.csv', "r") as csvfile:
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

#This function removes punctuation and numbers 
def clean_text(phrase):
    pattern = r'[0-9]'
    final = ""
    for i in phrase:
        if i not in punctuation: 
            new_word = re.sub(pattern, '', i)
            final += new_word
    return final 

def classifyMessage(input):
    cleaned_phrase = clean_text(input.lower())
    noStopWords = remove_stopwords(cleaned_phrase)
    
    phrase = tokenizer.texts_to_sequences([noStopWords.lower()])
    padded_phrase = pad_sequences(phrase, maxlen=50, truncating='post', padding='post')
    pred = model.predict(padded_phrase)
    pred_int = pred.round().astype("int")

    return pred_int


#This function corrects the spelling of a city  
def correct_spelling(user_response):
    city_names = list(cities.keys())
    response = user_response.title()
    split_string = re.split("\s|(?<!\d)[,.](?!\d)", response)
    city = ""
    match_score = 0
    for word in split_string:
        matches = difflib.get_close_matches(word, city_names)
        if matches:
            score = difflib.SequenceMatcher(None, word, matches[0]).ratio() 
            if score > match_score:
                match_score = score
                city = matches[0]
    if score <= 0.4:
        city = 'invalid'
    return city

#This function finds mental health resources in the user's vicinity using Yelp's API

url = 'https://api.yelp.com/v3/businesses/search'
key = open(r'/Users/gurnirmalkaur/Desktop/pathBASEF/key.txt').readlines()[0]

def find_resource(city, category):
    location = correct_spelling(str(city))
    if location == 'invalid':
        location = "Brantford"
        
    term = "mental health"
    resources = ""
    headers = {
    'Authorization': 'Bearer %s' % key
    }       
    parameters = {'location': location,
                'categories': category,
                'term': term,
                'limit': 3}
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

def yes_no(response):
    yes = ["yes", "yeah", "yep", "y" "sure", "ya", "yah", "yea"]
    no = ["no", "nah", "nope", "n",]

    if any(substring in response.lower() for substring in yes):
        answer = 'y'
    elif any(substring in response.lower() for substring in no):
        answer = 'n'
    else:
        answer = "Sorry, I don't understand"
    return answer

def check_answer1(msg):
    answer = yes_no(msg)
    if answer == "y":
        reply = "I know asking for help is hard, but if you are struggling, there are people who want to help you. Call 833-456-4566 to connect with responders available 24/7. This is a safe and confidential place to talk and get support.\n"
        reply += "If you don't feel like talking on the phone, can you tell me what city you are in so I can help you find some other resources?"
    elif answer == "n":
        reply = "Okay. Are you having feelings of anxiety or depression?"
    else:
        reply = "Sorry, I don't understand. Are you having suicidal thoughts?"
        answer = 'invalid'
    return reply, answer 

def check_answer2(response):
    answer = yes_no(response)
    if answer == "y":
        reply = "Can you tell me what city you're in? That way I can help you find some resources in your area."
    elif answer == "n":
        reply = "Okay, I'm glad to hear that. If you ever do need help, just say !Hera"
    else:
        answer = 'invalid'
        reply = "Sorry, I don't understand. Are you having feelings of anxiety or depression?"
    return reply, answer


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
            user = message.author
            if res[0][0] == 1:
            
                await message.reply("Hi, I noticed you may be experiencing emotional distress. If so, I want to help you. Are you having suicidal thoughts?")    
                invalid = True 

                while invalid:
                    msg = await client.wait_for("message",check = lambda m: m.author == user) 
                    reply, state = check_answer1(msg.content)
                    await msg.reply(reply)

                    if state == 'y':
                        city_msg = await client.wait_for("message",check = lambda m: m.author == user)
                        await city_msg.reply(find_resource(city_msg.content, "hospital"))

                    if state != 'invalid':
                        invalid = False
                
                invalid = True 
                while invalid:
                    response = await client.wait_for("message",check = lambda m: m.author == user)
                    reply, state = check_answer2(response.content)
                    await response.reply(reply)

                    if state != 'invalid':
                        invalid = False
                
                city = await client.wait_for("message",check = lambda m: m.author == user)
                await city.reply(find_resource(city.content, "c_and_mh"))
            
    client.run(token)



