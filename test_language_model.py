import pytest

import implementation.language_detector as language_detector

TOY_FILE = "data/train/en/dummy.txt"

TRAIN_FILE_EN = 'data/train/en/all_en.txt'
TRAIN_FILE_ES = 'data/train/es/all_es.txt'
TEST_FOLDER = 'data/test/'

def test_calculate_counts_unigrams():
    unigrams, bigrams = language_detector.calculate_counts(TOY_FILE)
    assert unigrams['a'] == 7
    assert unigrams['w'] == 1
    assert unigrams['$'] == 18
    assert len(unigrams) == 24

def test_calculate_counts_bigrams():
    unigrams, bigrams = language_detector.calculate_counts(TOY_FILE)
    assert bigrams['a']['l'] == 1
    assert bigrams['a']['b'] == 2
    assert bigrams['a']['s'] == 1
    assert bigrams['a']['m'] == 1
    assert bigrams['a']['p'] == 1
    assert bigrams['a']['y'] == 1

    assert bigrams['w']['x'] == 1

    assert bigrams['a']['a'] == 0
    assert bigrams['w']['a'] == 0


def test_calculate_probabilities_unsmoothed():
    unigrams, bigrams = language_detector.calculate_counts(TOY_FILE)
    bigram_prob = language_detector.calculate_probabilities(unigrams, bigrams, TOY_FILE,
                                                            smoothed=False, log=False)

    assert 0.14 == pytest.approx(bigram_prob['a']['l'], abs=0.01)
    assert 0.29 == pytest.approx(bigram_prob['a']['b'], abs=0.01)
    assert 0.14 == pytest.approx(bigram_prob['a']['s'], abs=0.01)
    assert 0.14 == pytest.approx(bigram_prob['a']['m'], abs=0.01)
    assert 0.14 == pytest.approx(bigram_prob['a']['p'], abs=0.01)
    assert 0.14 == pytest.approx(bigram_prob['a']['y'], abs=0.01)
    assert 1.00 == pytest.approx(bigram_prob['w']['x'], abs=0.01)
    assert 0.00 == pytest.approx(bigram_prob['a']['a'], abs=0.01)
    assert 0.00 == pytest.approx(bigram_prob['w']['a'], abs=0.01)


def test_calculate_probabilities_smoothed():
    unigrams, bigrams = language_detector.calculate_counts(TOY_FILE)
    bigram_prob = language_detector.calculate_probabilities(unigrams, bigrams, TOY_FILE,
                                                            smoothed=True, log=False)

    assert 0.06 == pytest.approx(bigram_prob['a']['l'], abs=0.01)
    assert 0.10 == pytest.approx(bigram_prob['a']['b'], abs=0.01)
    assert 0.06 == pytest.approx(bigram_prob['a']['s'], abs=0.01)
    assert 0.06 == pytest.approx(bigram_prob['a']['m'], abs=0.01)
    assert 0.06 == pytest.approx(bigram_prob['a']['p'], abs=0.01)
    assert 0.06 == pytest.approx(bigram_prob['a']['y'], abs=0.01)
    assert 0.08 == pytest.approx(bigram_prob['w']['x'], abs=0.01)


def test_calculate_log_prob():
    sents_probs = [("alphabet has 26 letters", -22.60),
                   ("this is a dummy sentence", -32.72),
                   ("blah blah", -14.81)]
    model = language_detector.create_model(TOY_FILE)
    for sent, prob in sents_probs:
        text = language_detector.preprocess(sent)
        probability = language_detector.calculate_log_prob(text, model)
        assert prob == pytest.approx(probability, abs=0.01)


def test_language_detector():
    accuracy = language_detector.predict_and_evaluate(TRAIN_FILE_EN, TRAIN_FILE_ES, TEST_FOLDER)
    assert accuracy == pytest.approx(1.0)
