import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df= pd.read_csv("C:/Users/admin/Desktop/NLP-Deployment-Heroku-master/spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
	
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
#    
pickle.dump(cv, open('tranform.pkl', 'wb'))
    
    
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,y)

filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
    
#Alternative Usage of Saved Model
# joblib.dump(clf, 'NB_spam_model.pkl')
# NB_spam_model = open('NB_spam_model.pkl','rb')
# clf = joblib.load(NB_spam_model)
