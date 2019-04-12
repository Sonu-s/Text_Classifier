# Importing the Librearies..
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

# Importing Dataset..
reviews = load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

# Storing as pickle file..
with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)
    
# Unpickling the dats set..
with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)
    
# Creating corpus
corpus = []
for i in range(0,len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ', review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    
# BOw..
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

#TF-IDF Model...
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting LOgisting Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Pickling the classifire
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

# Pickling the Vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

# Unpickling the classifier and vectorizer
with open('classifier.pickle','rb') as f:
    clf= pickle.load(f)

# Unpickling the classifier and vectorizer
with open('tfidfmodel.pickle','rb') as f:
    tfidf= pickle.load(f)
    
sample = ["Happy bday "]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))