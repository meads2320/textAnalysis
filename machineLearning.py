# Text Classification

# Identifying category or class of given text such as a blog, book, web page, news articles, and tweets

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt

#Loading Data
data=pd.read_csv('train.tsv', sep='\t')

data.head()

#This data has 5 sentiment labels:

# 0 - negative 1 - somewhat negative 2 - neutral 3 - somewhat positive 4 - positive
data.info()

data.Sentiment.value_counts()

# 2    79582
# 3    32927
# 1    27273
# 4     9206
# 0     7072
# Name: Sentiment, dtype: int64

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


#Feature Generation using Bag of Words

#we have a set of texts and their respective labels.
#But we directly can't use text for our model.

#We need to convert these text into some numbers or vectors of numbers.

#Bag-of-words model(BoW) is the simplest way of extracting features from the text

#Example: There are three documents:

#Doc 1: I love dogs.
#Doc 2: I hate dogs and knitting.
#Doc 3: Knitting is my hobby and passion.

#Create a matrix of document and words by counting the occurrence of words

#This matrix is using a single word.
#It can be a combination of two or more words, which is called a bigram or trigram model

#the general approach is called the n-gram model

#We can generate document term matrix by using scikit-learn's CountVectorizer.

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])

#Split train and test set
#To understand model performance, dividing the dataset into a training set and a test set is a good strategy.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)

#Model Building and Evaluation

#Let's build the Text Classification Model using TF-IDF.

#First, import the MultinomialNB module and create a Multinomial Naive Bayes classifier object using MultinomialNB() function.
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Then, fit your model on a train set using fit() and perform prediction on the test set using predict().
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

#Output: MultinomialNB Accuracy: 0.604916912299

# A classification rate of 60.49% using CountVector(or BoW) is not considered as good accuracy


#Feature Generation using TF-IDF

#In Term Frequency(TF), you just count the number of words occurred in each document.

#The main issue with this Term Frequency is that it will give more weight to longer documents

# IDF(Inverse Document Frequency) measures the amount of information a given word provides across the document.

# IDF is the logarithmically scaled inverse ratio of the number of documents that contain the word and the total number of documents.

#TF-IDF(Term Frequency-Inverse Document Frequency) normalizes the document term matrix

#the product of TF and IDF

# Word with high tf-idf in a document, it is most of the times occurred in given documents and must be absent in the other documents

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

#Output: MultinomialNB Accuracy: 0.586526549618
