import re
import os
import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


keywords = ["personal","projects","experience","skills","education"]
indexes = []
categories = {}
for w in keywords:
    categories[w] = []

stop_words = stopwords.words('english')
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    # resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = [x for x in resumeText.split() if x not in stop_words]     #Removing stopwords
    return resumeText

MAX_SEQUENCE_LENGTH = 100
MAX_WORDS = None


dir = os.listdir('./train_txt/')

# start preprocessing
for filename in dir:
    file = open('./train_txt/'+ filename,'r')
    read = file.read()
    read = read.lower()
    file.close()

    read = cleanResume(read)

    hash = {}
    hash["personal"] =0
    for word in keywords:
        if word in read:
            hash[word] = read.index(word)


    items = sorted(hash.items(), key = lambda x: x[1])
    for i in range(len(items)):
        start = items[i][1]
        end = None
        if (i+1)==len(items):
            end = len(read)
        else:
            end = items[i+1][1]

        categories[items[i][0]].append(read[start:end])

    for w in keywords:
        if w not in hash.keys():
            categories[w].append('None') 

    indexes.append(filename)


data = []
words_dict = ['None']
for i in range(len(indexes)):
    row = []
    for w in keywords:
        row.append(categories[w][i])
        if categories[w][i] is not  None:
            words_dict+=categories[w][i]

    data.append(row)

words_dict = set(words_dict)
MAX_WORDS = len(words_dict)+1
tokenizer  = Tokenizer(nb_words = MAX_WORDS)
tokenizer.fit_on_texts(words_dict)
sequences = tokenizer.texts_to_sequences(words_dict)
word_index = tokenizer.word_index
print(" found itna unique tokens ", len(word_index))


# df = pd.DataFrame(data, index=indexes, columns=keywords)


# df.to_csv('data.csv')
# print(categories['other'])






