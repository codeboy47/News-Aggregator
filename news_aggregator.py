#!/usr/bin/python


### load libraries
import pickle
from time import time
import pandas as pd
from sklearn.base import clone
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# The words (features) and classes (labels), already largely processed.
# These files have been created from parse_out_text
words_file = "tools/your_features.pkl"
target_file = "tools/your_labels.pkl"
features = pickle.load( open(words_file, "r"))
labels = pickle.load( open(target_file, "r"))



###############################################################################
### Preprocessing the dataset
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state = 42, shuffle = True)



###############################################################################
# Text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)
feature_names = vectorizer.get_feature_names()
print feature_names[:150]
print features_train.shape



"""
###############################################################################
# Apply dimensionality reduction using truncated SVD (aka LSA).
tsvd = TruncatedSVD(n_components = 1000, n_iter = 5, random_state = 42)
features_tsvd = tsvd.fit_transform(features)
print "\nweight of 1000 features : \n", tsvd.explained_variance_ratio_
print "\nNo of original features : ", features.shape[1]
print "No of principal components : ", features_tsvd.shape[1]
"""


###############################################################################
# Apply feature selection
### feature selection, because input is super high dimensional and
### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed  = selector.transform(features_test)
print "No of features after selection :", features_train_transformed.shape[1]



###############################################################################
# Identifying best parameters for our classifiers

parameters = [

    {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1, 10]
    },

    {
         'multi_class' : ('multinomial', 'ovr'),
         'solver' : ('newton-cg', 'sag', 'saga', 'lbfgs'),
         'C': [0.01, 0.1, 1, 10, 100]
    },

    {
         'criterion' : ('gini', 'entropy'),
         'n_estimators': [10, 50, 100, 150, 200],
         'max_features' : ('sqrt', 'log2'),
    },

    {
         'n_estimators': [30, 50, 80, 100],
         'learning_rate' : [1.5, 1, 0.7, 0.5]
    }
]


models = []
models.append(('MNB', MultinomialNB())) # use multinomial not gaussian as we have discrete data i.e. store count of each unique word
models.append(('LR', LogisticRegression(class_weight = 'balanced')))
models.append(('RF', RandomForestClassifier()))
models.append(('AB', AdaBoostClassifier()))


clfWithBestParameters1 = []
clfWithBestParameters2 = []

i = 0
for name, clf in models:

    print "\nFitting the classifier ", name, " to training dataset"
    t0 = time()
    classifier1 = GridSearchCV(clf, parameters[i])
    classifier1.fit(features_train, labels_train)
    print "done in : ", round(time() - t0,3), "s"
    print "best parameters selected for all training features : \n", classifier1.best_params_
    clfWithBestParameters1.append(clone(classifier1))

    print "\nFitting the classifier ", name, " to transformed dataset"
    t0 = time()
    classifier2 = GridSearchCV(clf, parameters[i])
    classifier2.fit(features_train_transformed, labels_train)
    print "done in : ", round(time() - t0,3), "s"
    print "best parameters selected for transformed features : \n", classifier2.best_params_
    clfWithBestParameters2.append(clone(classifier2))

    i += 1



###############################################################################
# Validation - Check for overfitting to the training set or a data leak

i = 0
highestAcc = 0
for name, clf in models:

    # estimate the accuracy of a model on the dataset by splitting the data, fitting a
    # model and computing the score 10 consecutive times (with 10 different splits each time)
    skf = model_selection.StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)

    # finding accuracy with all features
    classifier1 = clfWithBestParameters1[i]
    acc1 = model_selection.cross_val_score(classifier1, features_train, labels_train, cv = skf, scoring='accuracy')
    print "accuracy for ", name, " with training features : ", acc1.mean()

    # finding accuracy with transformed features
    classifier2 = clfWithBestParameters2[i]
    acc2 = model_selection.cross_val_score(classifier2, features_train_transformed, labels_train, cv = skf, scoring='accuracy')
    print "accuracy for ", name, " with transformed features : ", acc2.mean()

    if highestAcc < acc1.mean():
        highestAcc = acc1.mean()
        bestModel = clone(classifier1)

    i += 1


print "So the most accurate model is : ", bestModel



###############################################################################
# Final quantitative evaluation of the best model quality on the data set
# Make predictions on validation dataset

bestModel.fit(features_train, labels_train)
predictions = bestModel.predict(features_test)
print accuracy_score(labels_test, predictions)
print confusion_matrix(labels_test, predictions)
print classification_report(labels_test, predictions)
