#!/usr/bin/python


### load libraries
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from stemText import stem_text


### Load dataset
# https://www.kaggle.com/uciml/news-aggregator-dataset/home
# importing a csv file using Pandas
# takes csv file and loads into data frame
newsDF = pd.read_csv("../input/uci-news-aggregator.csv")


# let's take a look at our data
print newsDF.head(n = 10)


# pick only TITLE and CATEGORY columns only
# This will return a new data frame with 2 columns
newsDF = newsDF[['TITLE', 'CATEGORY']]


# modify the title column with stemmed text
newsDF['TITLE'] = [stem_text(title) for title in newsDF['TITLE']]

print "You ARE HERE !!", newsDF.head(n = 10)

features = newsDF['TITLE']


# Encode labels with value between 0 and n_classes-1 as labels are in form of characters
encoder = LabelEncoder()
labels = encoder.fit_transform(newsDF['CATEGORY'])
print list(encoder.inverse_transform([0,1,2,3])) # ans : ['b', 'e', 'm', 't']
print labels


pickle.dump( features, open("your_features.pkl", "w") )
pickle.dump( labels, open("your_labels.pkl", "w") )
