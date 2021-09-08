from nltk.corpus import stopwords
stop_words = stopwords.words('english')


last_index = len(stop_words) - 1
stopWords = stop_words[26:29]
stopWords.extend(stop_words[35:last_index])

print(stopWords)