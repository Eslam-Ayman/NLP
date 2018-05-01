from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import pickle
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
y = yelp['class']
print(X)
print(X.shape)
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)
 #لغاية هنا كدا دا كان جزء الpreprocessing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=101)
model = MultinomialNB()
model.fit(X_train, y_train)
filename = 'finalized_naive_bayes_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)