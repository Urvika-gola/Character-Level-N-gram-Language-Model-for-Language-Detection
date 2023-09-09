import argparse
import os
import random
import re
import collections
import math
import logging
logging.basicConfig(level=logging.INFO)


def preprocess(line):
    # DO NOT CHANGE THIS METHOD unless you want to explore other preprocessing
    #   options and discuss your findings in the report

    # get rid of the stuff at the end of the line (spaces, tabs, new line, etc.)
    line = line.rstrip()
    # lower case
    line = line.lower()
    # remove everything except characters and white space
    line = re.sub("[^a-z ]", '', line)
    # tokenize; not done "properly" but sufficient for now
    tokens = line.split()

    # adding $ before and after each token because we are working with bigrams
    tokens = ['$' + token + '$' for token in tokens]

    return tokens


def calculate_counts(path):
    # This is just some Python magic ...
    # unigrams[key] will return 0 if the key doesn't exist
    unigrams = collections.defaultdict(int)
    # and then you have to figure out what bigrams and bigram_prob will return
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))

    f = open(path, 'r')
    # You shouldn't visit a token more than once
    # Store in unigrams and bigrams the counts of unigrams and bigrams in path
    for l in f.readlines():
        tokens = preprocess(l)
        # YOUR CODE GOES HERE


    return unigrams, bigrams


def calculate_probabilities(unigrams, bigrams, path, smoothed=False, log=False):
    bigram_prob = collections.defaultdict(lambda: collections.defaultdict(float))
    # return raw probabilities by default
    # return smoothed probabilities (add-one smoothing) if smoothed is set to True
    # return log (of unsmoothed or smoothed) probabilities if log is set to True

    # YOUR CODE GOES HERE
    return bigram_prob


def create_model(path):
    # DO NOT CHANGE THIS METHOD
    unigrams, bigrams = calculate_counts(path)
    # the "good" model will use log smoothed probabilities
    bigram_prob = calculate_probabilities(unigrams, bigrams, str(path), smoothed=True, log=True)

    return unigrams, bigram_prob


def calculate_log_prob(text, model):
    unigrams, bigram_prob = model

    log_prob = 0
    # return the log probability of text according to model; smooth on the fly unseen bigrams (it is faster)

    # YOUR CODE GOES HERE
    return log_prob


def predict(file, model_en, model_es):
    f = open(file, 'r')
    text = []
    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue
        text.extend(tokens)

    # Figure out the language after calculating the log probabilities
    prob_en = calculate_log_prob(text, model_en)
    prob_es = calculate_log_prob(text, model_es)

    lang = "unk"
    # YOUR CODE GOES HERE

    return lang


def predict_and_evaluate(en_tr, es_tr, folder_te):
    # DO NOT CHANGE THIS METHOD

    # STEP 1: create a model for English with en_tr
    model_en = create_model(en_tr)

    # STEP 2: create a model for Spanish with es_tr
    model_es = create_model(es_tr)

    for sent in ["alphabet has 26 letters", "this is a dummy sentence", "blah blah"]:
        logging.debug(f"Log probability of '{sent}': {calculate_log_prob(preprocess(sent), model_en):.2f}")

    # STEP 3: loop through all the files in folder_te and make predictions
    predictions_correct = []
    for lang in ["en", "es"]:
        folder = os.path.join(folder_te, lang)
        logging.info(f"Prediction for *{lang}* documents:")
        for f in os.listdir(folder):
            f_path = os.path.join(folder, f)
            pred = predict(f_path, model_en, model_es)
            predictions_correct.append(pred == lang)
            logging.info(f"{f}\t{pred}")
        logging.info("")

    return predictions_correct.count(True) / len(predictions_correct)


def main(en_tr, es_tr, folder_te):
    # DO NOT CHANGE THIS METHOD

    accuracy = predict_and_evaluate(en_tr, es_tr, folder_te)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    # DO NOT CHANGE THIS CODE

    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR_EN",
                        help="Path to file with English training files")
    parser.add_argument("PATH_TR_ES",
                        help="Path to file with Spanish training files")
    parser.add_argument("PATH_TEST",
                        help="Path to folder with test files")
    args = parser.parse_args()

    main(args.PATH_TR_EN, args.PATH_TR_ES, args.PATH_TEST)
