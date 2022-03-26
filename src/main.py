"""Module to test the classifier in console"""
import networkx as nx

from negation import get_cue_pred, get_scope_pred, load_clsf
from utils import get_idx_cues, get_sent_feat, reformat_prediction
from negation import nlp


def get_input():
    "TESTING: to get input sentences and detect the cues and its scope"
    cue_clf, cue_vect, scope_clf, scope_vect = load_clsf()
    sent = input('Please enter a sentence: ')
    features = get_sent_feat(sent)
    cue_pred = get_cue_pred(features, cue_clf, cue_vect)
    scope_pred = get_scope_pred(features, get_idx_cues(cue_pred), scope_clf, scope_vect)
    prediction = reformat_prediction(features, cue_pred, scope_pred)
    return prediction


if __name__ == '__main__':
    get_input()