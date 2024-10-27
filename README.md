# N-gram Language Model for Language Detection

## Overview
This project implements a character-level bigram language model for **language detection**. Given a text, the model predicts whether it is written in **English** or **Spanish**. It utilizes n-grams and **add-one smoothing** to handle unseen n-grams, with log probabilities to prevent underflow.

Developed for **CSC 585: Algorithms for NLP** under **Instructor Eduardo Blanco** at the **University of Arizona**.

## Features

- **Character-Level Bigram Model**: Trains on character sequences, considering the probability of each character following another within each language.
- **Add-One Smoothing**: Applies add-one smoothing to account for unseen bigrams in both training and prediction phases.
- **Log Probability Calculations**: Uses log probabilities to handle floating-point precision and avoid zero probabilities.
- **Efficient Language Prediction**: Predicts the language of text files in less than 1 minute.

### Implementation Details
- Unigram and Bigram Counting: The create_model function tokenizes each line and calculates unigram and bigram counts for each character.

- Probability Calculation:
Uses add-one smoothing when smoothed=True.
Returns log probabilities when log=True.

- Language Prediction:
Calculates the log probability of text according to the model using calculate_log_prob.
Determines language by comparing the log probabilities for English and Spanish.

## Assignment Requirements
This project fulfills the following requirements:

- Character-Level Bigram Modeling: Character n-grams are used to distinguish between English and Spanish text.
- Smoothing and Log Probability: Add-one smoothing is applied, and log probabilities are used.

## Evaluation:
- Includes functions to evaluate counts and probabilities using public test cases.
- Provides predictions based on language models for English and Spanish.
