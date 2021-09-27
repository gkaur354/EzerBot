"""
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
last_index = len(stop_words) - 1
stopWords = stop_words[26:29]
stopWords.extend(stop_words[35:last_index])
last_index = len(stop_words) - 1
stopWords = stop_words[4:last_index]

print(stop_words)
"""
import csv
import collections
from types import CoroutineType
from typing import DefaultDict
cities = DefaultDict(list)
with open('/Users/gurnirmalkaur/Desktop/canadaCities.csv', "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        city = row[1]
        province = row[0]
        cities[city].append(province)

import difflib
city_names = list(cities.keys())

user_response = "taranto"
response = user_response.capitalize()

if city_names.count(response):
    print(response)
else:
    matches = difflib.get_close_matches(response, city_names)
    print("Did you mean",matches[0],"?")
