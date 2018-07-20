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
| Algorithm | Total features | Transformed features |
| --- | Best Parameters | Training time | Accuracy | | Best Parameters | Training time | Accuracy |
| Naive Bayes Classifier | 1.745 | 0.188 | 0.973265073948 | 1.745 | 0.188 | 0.973265073948 |


<br>

## Conclusions:



<br>

## Result:
Logistic regression gives highest accuracy of 94.29% when we use StratifiedKFold on training dataset. It gives 94.15% accuracy when we run on testing dataset.
