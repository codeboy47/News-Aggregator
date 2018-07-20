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
1. Dataset is loaded by importing a csv file using Pandas and load into data frame.
2. Stop words are removed from title and stemming is performed to reduce a word to its stem. Then a string containing important information about title is returned.
3. Labels are encoded with value between 0 and n_classes-1 as they are in form of characters
4. These features and labels are stored in a pickle file.
5. After loading the pickle files, features and labels are split into training and testing data.
6. Now training and testing features are pulled into vectors using TfidfVectorizer.
7. Due to high dimensional input, feature selection is applied on training and testing features which selects 10% of features that are most powerful.
8. GridSearchCV is used to identify best parameters for our classifiers.
9. For validation purpose, overfitting to the training set or a data leak is checked. This is acheived by using cross_val_score that estimates the accuracy of a model on the dataset by splitting the data, fitting a model and computing the score 10 consecutive times (with 10 different splits each time). 
10. Final quantitative evaluation of best model is done on testing dataset.

Note : Here we apply cross_val_score on both training and transformed features(features after using feature selection). 

<br>

## Observation:

Table for total features:

| Classifier | Best Parameters | Training time(s) | Accuracy(%)  |
| --- | --- | --- | --- |
| Multinomial Naive Bayes | alpha = 0.1 | 0.116 | 92.7 |
| Logistic regression | C = 1, multi_class = multinomial, solver = newton-cg | 23.688 | 94.29 | 	
| AdaBoost | n_estimators = 80, learning_rate = 0.5  | 21534.235 |88.62 |	
| Random Forest | criterion = gini, n_estimators = 100, max_features = sqrt | 37419.736 | 92.13 |

Table for transformed features:

| Classifier | Best Parameters | Training time(s) | Accuracy(%)  |
| --- | --- | --- | --- |
| Multinomial Naive Bayes | alpha = 0.1 | 0.108 | 91.31 |
| Logistic regression | C = 1, multi_class = multinomial, solver = newton-cg | 14.739 | 92.79 | 
| AdaBoost | n_estimators = 50, learning_rate = 1 | 34.28 | 59.75 |
| Random Forest | criterion = gini, n_estimators = 10, max_features = sqrt | 119.375 | 92.01 |
<br>

## Conclusions:
1. Accuracy for random forest on both types of data is approximately same but it takes almost 10 hours to train total features whereas it takes just 2 minutes for transformed data.
2. Multinomial Naive Bayes and Logistic regression have same parameters for both types of features.
3. Accuracy for AdaBoost increases significantly if we increase number of features but training time of total features is huge.
4. Accuracy for Logistic regression increases by 2% when we train on entire training dataset as the training time difference between 2 types of features is only 9 seconds. 

<br>

## Result:
Logistic regression gives highest accuracy of 94.29% when we use StratifiedKFold on training dataset. It gives 94.15% accuracy when we run on testing dataset.
