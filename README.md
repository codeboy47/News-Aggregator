# New-Aggregator
<br>

## Objective:
To predict the category of news article on the basis of title.

<br>

## Dataset:
The data is the UCI News Aggregator dataset (https://archive.ics.uci.edu/ml/datasets/News+Aggregator) which contains headlines, URLs, and categories for 422,937 news stories collected by a web aggregator between March 10th, 2014 and August 10th, 2014.
News categories included in this dataset include business; science and technology; entertainment; and health. Different news articles that refer to the same news item (e.g., several articles about recently released employment statistics) are also categorized together.

<br>

## Description:
1. 

<br>

## Observation:

| Classifier | Best Parameters | Training time(s) | Accuracy(%)  |
| --- | --- | --- | --- |
| Multinomial Naive Bayes | alpha = 0.1 | 0.116 | 92.7 |


Table for transformed features:
| Classifier | Best Parameters | Training time(s) | Accuracy(%)  |
| --- | --- | --- | --- |
| Multinomial Naive Bayes | alpha = 0.1 | 0.108 | 91.31 |
| Logistic regression | C = 1, multi_class = multinomial, solver = newton-cg | 14.739 | 92.79 | 
| AdaBoost | n_estimators = 50, learning_rate = 1 | 34.28 | 59.75 |
| Random Forest | criterion = gini, n_estimators = 10, max_features = sqrt | 119.375 | 92.01 |
<br>

## Conclusions:



<br>

## Result:
Logistic regression gives highest accuracy of 94.29% when we use StratifiedKFold on training dataset. It gives 94.15% accuracy when we run on testing dataset.
