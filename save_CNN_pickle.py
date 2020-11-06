import pandas as pd
import pandas as pd
import numpy as np
import pickle
import swifter
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
import re  # regular expressions
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Load model, and tf_transformer and count_vec from pickle files

model_IE = pickle.load(open('mbti_IE_lr.pickle', 'rb')) #LR
model_NS = pickle.load(open('mbti_NS_lr.pickle', 'rb')) #LR
model_TF = pickle.load(open('mbti_TF_lr.pickle', 'rb')) #LR
model_PJ = pickle.load(open('mbti_PJ_lr.pickle', 'rb')) #LR

tf_transformer_IE = pickle.load(open('mbti_IE_tf.pickle', 'rb')) #TF
tf_transformer_NS = pickle.load(open('mbti_NS_tf.pickle', 'rb')) #TF
tf_transformer_TF = pickle.load(open('mbti_TF_tf.pickle', 'rb')) #TF
tf_transformer_PJ = pickle.load(open('mbti_PJ_tf.pickle', 'rb')) #TF

count_vec_IE = pickle.load(open('mbti_IE_bow.pickle', 'rb')) #FM
count_vec_NS = pickle.load(open('mbti_NS_bow.pickle', 'rb')) #FM
count_vec_TF = pickle.load(open('mbti_TF_bow.pickle', 'rb')) #FM
count_vec_PJ = pickle.load(open('mbti_PJ_bow.pickle', 'rb')) #FM



path = 'C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_NLP\\sent_mails_.csv'
df = pd.read_csv(path,index_col=None)

print(df.head())
print(df.columns)



def mbtipredict_IE(x):
    language = 'english'
    minWordSize = 3
    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(x))

    # convert to lower-case
    text_lower = text_alpha_chars.lower()

    # remove stop words
    stops = set(stopwords.words(language))
    text_no_stop_words = ' '
    # print(text_no_stop_words)
    for w in text_lower.split():
        if w not in stops:
            text_no_stop_words = text_no_stop_words + w + ' '
            # print(text_no_stop_words)
    # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '
        # print(text_stemmer)

    # remove short words
    processedtext = ' '
    for w in text_stemmer.split():

        if len(w) >= minWordSize:
            processedtext + w + ' '
            #print(processedtext)
            processedtext = [processedtext]
            #print(processedtext)
            X_bag_of_words = count_vec_IE.transform(processedtext)
            #print(X_bag_of_words)
            X_tf = tf_transformer_IE.transform(X_bag_of_words)
            #print(X_tf)
            mbtipredict = model_IE.predict(X_tf)

            return mbtipredict


def mbtipredict_NS(x):
    language = 'english'
    minWordSize = 3
    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(x))

    # convert to lower-case
    text_lower = text_alpha_chars.lower()

    # remove stop words
    stops = set(stopwords.words(language))
    text_no_stop_words = ' '
    # print(text_no_stop_words)
    for w in text_lower.split():
        if w not in stops:
            text_no_stop_words = text_no_stop_words + w + ' '
            # print(text_no_stop_words)
    # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '
        # print(text_stemmer)

    # remove short words
    processedtext = ' '
    for w in text_stemmer.split():

        if len(w) >= minWordSize:
            processedtext + w + ' '
            #print(processedtext)
            processedtext = [processedtext]
            #print(processedtext)
            X_bag_of_words = count_vec_NS.transform(processedtext)
            #print(X_bag_of_words)
            X_tf = tf_transformer_NS.transform(X_bag_of_words)
            #print(X_tf)
            mbtipredict = model_NS.predict(X_tf)

            return mbtipredict


def mbtipredict_TF(x):
    language = 'english'
    minWordSize = 3
    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(x))

    # convert to lower-case
    text_lower = text_alpha_chars.lower()

    # remove stop words
    stops = set(stopwords.words(language))
    text_no_stop_words = ' '
    # print(text_no_stop_words)
    for w in text_lower.split():
        if w not in stops:
            text_no_stop_words = text_no_stop_words + w + ' '
            # print(text_no_stop_words)
    # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '
        # print(text_stemmer)

    # remove short words
    processedtext = ' '
    for w in text_stemmer.split():

        if len(w) >= minWordSize:
            processedtext + w + ' '
            #print(processedtext)
            processedtext = [processedtext]
            #print(processedtext)
            X_bag_of_words = count_vec_TF.transform(processedtext)
            #print(X_bag_of_words)
            X_tf = tf_transformer_TF.transform(X_bag_of_words)
            #print(X_tf)
            mbtipredict = model_TF.predict(X_tf)

            return mbtipredict


def mbtipredict_PJ(x):
    language = 'english'
    minWordSize = 3
    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(x))

    # convert to lower-case
    text_lower = text_alpha_chars.lower()

    # remove stop words
    stops = set(stopwords.words(language))
    text_no_stop_words = ' '
    # print(text_no_stop_words)
    for w in text_lower.split():
        if w not in stops:
            text_no_stop_words = text_no_stop_words + w + ' '
            # print(text_no_stop_words)
    # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '
        # print(text_stemmer)

    # remove short words
    processedtext = ' '
    for w in text_stemmer.split():

        if len(w) >= minWordSize:
            processedtext + w + ' '
            #print(processedtext)
            processedtext = [processedtext]
            #print(processedtext)
            X_bag_of_words = count_vec_PJ.transform(processedtext)
            #print(X_bag_of_words)
            X_tf = tf_transformer_PJ.transform(X_bag_of_words)
            #print(X_tf)
            mbtipredict = model_PJ.predict(X_tf)

            return mbtipredict

###################################################################################################
#df = df[0:10] # to predict on smaller test set

#df['fraudulant_mail'] = df['Content'].swifter.apply(lambda x: spampredict(x))
df['personality_IE'] = df['Content'].swifter.apply(lambda x: mbtipredict_IE(x))
df['personality_NS'] = df['Content'].swifter.apply(lambda x: mbtipredict_NS(x))
df['personality_TF'] = df['Content'].swifter.apply(lambda x: mbtipredict_TF(x))
df['personality_PJ'] = df['Content'].swifter.apply(lambda x: mbtipredict_PJ(x))
#df.loc[:,'personality_IE'] = df.groupby('Origin')['Content'].apply(lambda x: mbtipredict(x))
#df['personality_IE'] = df.groupby('Origin')['Content'].apply(lambda x: mbtipredict(x))

print(df.head(10))

df.to_csv('mbti_enron.csv', sep=',', encoding='utf-8')

#file = pd.read_csv('mbti_enron.csv')
#print(file.head(10))
#print(file.columns)





