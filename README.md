# Toxic-Comment-Classifier
This project implements a machine learning model for detecting toxicity in tweets. The model uses Natural Language Processing (NLP) techniques to preprocess text data, convert it into numerical features using TF-IDF vectorization, and classify tweets as toxic or non-toxic using a Naive Bayes classifier.

## Table of Contents
Overview
Usage
Model Details
Evaluation
Saving and Loading the Model

## Overview

The project aims to classify tweets into toxic and non-toxic categories. It involves several steps:

Data preprocessing using NLTK.
Text vectorization using TF-IDF.
Training a Naive Bayes classifier.
Evaluating the model using the ROC-AUC score.
Saving the trained model and TF-IDF vectorizer for future use.

## Usage

### Load the Dataset:
The dataset is loaded from a CSV file, and unnecessary columns are dropped during preprocessing.

### Text Preprocessing:
Tweets are cleaned and lemmatized using NLTK. This process involves removing non-alphabetical characters, tokenizing the text, and converting words to their base forms through lemmatization.

### TF-IDF Vectorization:
The cleaned tweets are transformed into numerical features using TF-IDF vectorization, which measures the importance of words in the context of the dataset.

### Model Training:
A Naive Bayes classifier is trained on the TF-IDF features. The dataset is split into training and testing sets to evaluate the model's performance.

### Model Prediction:
After training, the model can predict the toxicity of new tweets, providing both the predicted class (toxic or non-toxic) and a probability score.

## Model Details
Text Preprocessing:
The text data is cleaned and lemmatized using NLTK’s tools to prepare it for model training.

### Vectorization:
The text data is converted into numerical features using TF-IDF vectorization, which reflects the importance of each word within the dataset.

### Classifier:
A Multinomial Naive Bayes model is used for the binary classification of tweets as toxic or non-toxic.

## Evaluation
The model's performance is evaluated using the ROC-AUC score, a metric that measures the classifier's ability to distinguish between the toxic and non-toxic classes. A high ROC-AUC score indicates good model performance.

## Saving and Loading the Model
The trained model and the TF-IDF vectorizer are saved using Pickle. This allows the model and vectorizer to be loaded and used later without needing to retrain the model.
