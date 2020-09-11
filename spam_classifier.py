# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:15:58 2019

@author: TXF10LQ
"""


###IMPORT DATA
import pandas as pd
csv_file = r'C:\Users\TXF10LQ\OneDrive - The Home Depot\Documents\ML\ml_book_giron\spam_classifier.csv'
data = pd.read_csv(csv_file,encoding='latin-1')

dataset = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"text"})
dataset.head()


###DESCRIBE DATASET 
dataset.describe()
###MORE HAM THAN SPAM
dataset.groupby('label')['text'].count()

###convert label to a numerical variable
dataset['numerical_label'] = dataset.label.map({'ham':0, 'spam':1})
dataset.head()

###plot pie chart
import matplotlib.pyplot as plt
dataset["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

#top messages
import numpy as np
topMessages = dataset.groupby("text")["label"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)

###

#Text Preprocessing
#nltk.download("all")
##from nltk.corpus import nltk.download('punkt')
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

spam_messages = dataset[dataset["label"] == "spam"]["text"]
ham_messages = dataset[dataset["label"] == "ham"]["text"]

spam_words = []
ham_words = []

###Remove stop words like the will or you

def extractSpamWords(spamMessages):
    global spam_words
    words = [word.lower() for word in word_tokenize(spamMessages) if word.lower().isalpha() and word.lower() not in stopwords.words("english")]
    spam_words = spam_words + words
    
def extractHamWords(hamMessages):
    global ham_words
    words = [word.lower() for word in word_tokenize(hamMessages) if  word.lower().isalpha() and word.lower() not in stopwords.words("english")]
    ham_words = ham_words + words

spam_messages.apply(extractSpamWords)
ham_messages.apply(extractHamWords)


###Display top words
from wordcloud import WordCloud

###Spam world cloud
spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam_words))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


###Ham word cloud
ham_wordcloud = WordCloud(width=600, height=400).generate(" ".join(ham_words))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()




###Top ham word
ham_words = np.array(ham_words)
print("Top 10 Ham words are :\n")
pd.Series(ham_words).value_counts().head(n = 10)

###Top spam word
spam_words = np.array(spam_words)
print("Top 10 Spam words are :\n")
pd.Series(spam_words).value_counts().head(n = 10)

###length of the message
dataset["messageLength"] = dataset["text"].apply(len)
dataset["messageLength"].describe()

###Text transformation
###Lets clean our data by removing punctuations/ stopwords and stemming words¶
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

def cleanText(message):
    
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    
    return " ".join(words)

dataset["text"] = dataset["text"].apply(cleanText)
dataset.head(n = 10)   

###Lets convert our clean text into a representation that a machine learning model can understand. I'll use the Tfifd for this
###For each row Tfidf compute the tf-idf score for each single word - look at the link on fav to understand better
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(dataset["text"])
print(features.shape)


###Encode the ham/span
def encodeCategory(cat):
    if cat == "spam":
        return 1
    else:
        return 0
        
dataset["label"] = dataset["label"].apply(encodeCategory)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, dataset["label"], stratify = dataset["label"], test_size = 0.2)




###Create model - Naive Bayes Classifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score

from sklearn.naive_bayes import MultinomialNB
gaussianNb = MultinomialNB()
gaussianNb.fit(X_train, y_train)

y_pred = gaussianNb.predict(X_test)

print(fbeta_score(y_test, y_pred, beta = 0.5))


###Model not optimized - could be way better not the only classifier 