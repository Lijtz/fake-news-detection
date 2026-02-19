Fake News Detection System (Logistic Regression + TF-IDF)

#Overview

This project implements a machine learning pipeline to detect fake news articles using Natural Language Processing techniques.

#Tech Stack

Python

Scikit-learn

NLTK

TF-IDF

Logistic Regression

Matplotlib

#Dataset

ISOT Fake and Real News Dataset (~44,000 articles)

#Pipeline

Data cleaning (lowercasing, regex filtering)

Stopword removal

TF-IDF vectorization (unigram + bigram)

Logistic Regression training

Model evaluation (Accuracy: 99%)

Confusion matrix visualization

Model serialization

CLI inference system

#Results

Accuracy: 99%

Precision & Recall: ~0.99

Balanced classification performance

#How to Run

python train.py

python predict.py

pip install -r requirements.txt
