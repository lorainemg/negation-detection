"""Module to test the classifier in console"""
import networkx as nx

from negation import get_cue_pred, get_scope_pred, get_sent, load_clsf
from utils import get_idx_cues
from negation import nlp


def get_input():
    """TESTING: to get input sentences and detect the cues and its scope"""
    cue_clf, cue_vect, scope_clf, scope_vect = load_clsf()
    instances = get_sent()
    cue_pred = get_cue_pred(instances, cue_clf, cue_vect)
    scope_pred = get_scope_pred(instances, get_idx_cues(cue_pred), scope_clf, scope_vect)
    res = []
    for sent, c_p, s_p in zip(instances, cue_pred, scope_pred):
        for idx, c, s in zip(sent, c_p, s_p):
            word = sent[idx]['wd']
            if c == 1:
                res.append((word, 'C'))
            elif s == 'I':
                res.append((word, 'I'))
            else:
                res.append((word, 'O'))
    print(res)
    return res
    
    
def get_sent():
    "Get a sent from the user and extracts its features"
    s = input('Please enter a sentence: ')
    doc = nlp(s)
    edges = []
    instances = []
    for sent in doc.sents:
        sentence = {}
        len = 0
        for token in sent:
            for child in token.children:
                edges.append((token.i, child.i))
            feat = {'head': token.head.i, 'dep' : token.dep_, 'pos' : token.pos_, 'lem' : token.lemma_, 'wd': token.text }
            len += 1
            sentence[token.i] = feat
        sentence['length'] = len
        sentence['undir-graph'] = nx.Graph(edges)
        sentence['dir-graph'] = nx.DiGraph(edges)
        instances.append(sentence)
    return instances


if __name__ == '__main__':
    get_input()