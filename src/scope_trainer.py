"""Module to train scope classifier"""
#import eli5
from sklearn_crfsuite import CRF
import numpy
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from utils import get_cue_dep_path, get_head_cue_path


def train_scope_learner(sentences, prediction):
    "Trains the scope classifier. Returns an instance of the classifier"
    X = extract_scope_features(sentences, prediction)
    y = get_labels(sentences, prediction)
    vect = DictVectorizer()
    X = vect.fit_transform(X)
    # clsf = LogisticRegression(penalty='l2')# LinearSVC()
    # clsf = CRF()
    clsf = LinearSVC()
    # test_params(X, y, 'I')
    clsf.fit(X, y)
    return clsf, vect


def extract_scope_features(sentences, prediction):
    """
    Extracts features for the scope classifier from the sentence.
    Returns a list of features dictionaries and a list of labels
    """
    instances = []
    for sent, cues in zip(sentences, prediction):
        graph, digraph = sent['undir-graph'], sent['dir-graph']
        for cue in cues:
            # sent_feat = []
            for key, value in sent.items():
                features = {}
                if isinstance(key, int):
                    # Lexical features:
                    features['token'] = value['wd'].lower() if 'wd' in value else 'null'
                    features['lemma'] = value['lem'].lower()
                    features['pos'] = value['pos']


                    if key > 0:
                        before = sent[key-1]
                        features['before-bigram1'] = before['lem'].lower()
                        features['before-bigram2'] = before['pos']
                    if (key+1) in sent:
                        forward = sent[key+1]
                        features['forward-bigram1'] = forward['lem'].lower()
                        features['forward-bigram2'] = forward['pos']
                    # len = sent['length']
                    # features['place_cue'] = numpy.round(cue/len, 2)
                    # features['place_tok'] = numpy.round(key/len, 2)

                    # Cue features:
                    features['pos-cue'] = sent[cue]['pos']
                    # features['lemma-cue'] = sent[cue]['lem']
                    # features['dep-cue'] = sent[cue]['dep']

                    dist = key - cue
                    # features['dist'] = dist
                    if dist < 0:
                        if abs(dist) <= 7:
                            features['dist-cue-left'] = 'close'
                        else:
                            features['dist-cue-left'] = 'far'
                    elif dist > 0:
                        if abs(dist) <= 12:
                            features['dist-cue-right'] = 'close'
                        else:
                            features['dist-cue-right'] = 'far'
                    else:
                        features['dist-cue-left'] = '0'
                        features['dist-cue-right'] = '0'

                    # Syntactic features:
                    features['dep'] = value['dep']
                    path, length = get_cue_dep_path(graph, sent, cue, key)
                    features['dep-cue-path'] = path
                    features['path-len'] = length
                    path, length = get_head_cue_path(digraph, sent, cue, key)
                    features['head-cue-path'] = path
                    features['head-cue-len'] = length
                    instances.append(features)
                    # sent_feat.append(features)
            # instances.append(sent_feat)
    return instances


def get_labels(sentences, prediction):
    """
     Extracts labels for training the scope classifier.
     Label values: In-scope: I, Out-of-scope: O, Beginning-of-scope: B, Cue: C
     Returns a list of labels
    """
    # sent_lab = []
    labels = []
    for sent, pred in zip(sentences, prediction):
        for cues in pred:
            scp_idx = get_scope(sent['scopes'], cues)
            # prev_label = 'O'
            if scp_idx is None:
                labels.extend(['O' for key in sent if isinstance(key, int)])
            else:
                for key, val in sent.items():
                    if isinstance(key, int):
                        scope = sent['scopes'][scp_idx]
                        if any(key in s for s in scope):
                            # if prev_label == 'O':
                            #     labels.append('B')
                            #     prev_label = 'B'
                            # elif any(key == cue for cue in pred):
                            #     labels.append('C')
                            #     prev_label = 'I'
                            # else:
                            labels.append('I')
                            # prev_label = 'I'
                        else:
                            labels.append('O')
                            prev_label = 'O'
            # sent_lab.append(labels)
    return labels #sent_lab


def get_scope(scopes, c_idx):
    "Gets the scope of an index"
    for idx, scp in enumerate(scopes):
        if any(c_idx == word[1] for word in scp):
            return idx
    return None

