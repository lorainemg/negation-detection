import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from utils import *


def train_cue_learner(sentences):
    """Trains the cue classifier. Returns an instance of the trained classifier"""
    features = extract_cue_features(sentences)
    y = get_labels_cue(sentences)
    vect = DictVectorizer()
    X = vect.fit_transform(features)
    clf = LogisticRegression(penalty='l1', max_iter=150) # LinearSVC(C=C_Value)
    clf.fit(X, y)
    # test_params(X, y, pos_label=1)
    return clf, vect


def extract_cue_features(sentences):
    """
    Extracts features for the cue classifier from the sentence.
    Returns a list of features dictionaries and a list of labels, identifying if a word is a cue or not
    """
    instances = []
    for sent in sentences:
        for key, value in sent.items():
            features = {}
            if isinstance(key, int):
                # if not known_cue_lexicon(value, cue_lexicon):
                #     sent[key]['not-pred-cue'] = True
                #     continue

                word = value['wd'].lower() if 'wd' in value else value['lem'].lower()
                features['token'] = word
                features['lemma'] = value['lem'].lower()
                features['pos'] = value['pos']

                if key > 0:
                    prev_sent = sent[key-1]
                    features['before-bigram1'] = prev_sent['lem'].lower()
                    features['before-pos1'] = prev_sent['pos']
                if key > 2:
                    features['before-bigram2'] = sent[key-2]['lem'].lower()
                if (key+1) in sent:
                    next_sent = sent[key+1]
                    features['forward-bigram1'] = next_sent['lem'].lower()
                    features['forward-pos1'] = next_sent['pos']
                if (key+2) in sent:
                    features['forward-bigram2'] = sent[key+2]['lem'.lower()]
                instances.append(features)
    return instances


def get_cue_lexicon(sentences):
    """Returns a list of all simples cues in the data"""
    cue_lexicon = []
    for sent in sentences:
        for cues in sent['cues']:                     # gets the cues for each scope
            for c in cues:
                neg = c[0].lower()
                if neg not in cue_lexicon:
                    cue_lexicon.append(neg)
    return cue_lexicon


def get_labels_cue(sentences):
    """
    Extracts labels for training the cue classifier.
    Skip the not known cue words. For known cue words, label 1 means cue and 0 not cue
    """
    labels = []
    for sent in sentences:
        for key, value in sent.items():
            if isinstance(key, int):
                # if 'not-pred-cue' in value:
                #     continue
                if is_known_cue(key, sent):
                    labels.append(1)
                else:
                    labels.append(-1)
    return labels


def is_known_cue(key, sent):
    """Determines if a certain word is a marked cue"""
    for cues in sent['cues']:
        for cue in cues:
            if key == cue[1] and cue[2]:
                return True
    return False


def known_cue_lexicon(word, cue_lexicon):
    """"Determines if a token is a known cue"""
    tk = word['wd'] if 'wd' in word else word['lem']
    return tk.lower() in cue_lexicon
