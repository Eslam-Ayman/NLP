from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pickle
from sklearn import metrics
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
yelp = pd.read_csv('movie-pang02.csv')
X = yelp['text']
z=X
y = yelp['class']
print(X.shape)
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)
 #لغاية هنا كدا دا كان جزء الpreprocessing
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=101)
loaded_model = pickle.load(open('finalized_naive_bayes_model.sav', 'rb'))
result = loaded_model.predict(X_test[10])
print(result)
