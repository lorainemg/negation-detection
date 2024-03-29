"""Module with helper functions"""
import networkx as nx
import spacy
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from typing import List, Dict, Union
from sklearn.model_selection import GridSearchCV
from scipy import stats

nlp = spacy.load('es_core_news_sm')


def make_labels_cue(sentences: List[Dict[Union[int, dict], str]], labels: List[str]) -> List[str]:
    """Make labels for the words that weren't predicted by the classifier"""
    y = []
    i = 0
    for sent in sentences:
        sent_labels = []
        for key, value in sent.items():
            if isinstance(key, int):
                if 'not-pred-cue' in value:
                    sent_labels.append(-1)
                else:
                    sent_labels.append(labels[i])
                    i += 1
        y.append(sent_labels)
    return y


def make_labels_scope(sentence: List[Dict[Union[int, dict], str]], prediction: List[int], labels: List[str]) -> List[str]:
    """Make labels for all the sentences predicted by the classifier"""
    y = []
    count = 0
    i = 0
    for pred, sent in zip(prediction, sentence):
        if len(pred) == 0:
            y.append(['O' for key in sent if isinstance(key, int)])
        else:
            sent_label = []
            length = sent['length']
            for c in pred:
                if count >= len(labels):
                    break
                sent_label.append(normalize(labels[i:i+length]))
                i += length
                # count += 1
            y.append(join_prediction(sent_label))
    return y


def join_prediction(predictions: List[str]) -> str:
    res = predictions[0]
    for pred in predictions[1:]:
        for i in range(len(pred)):
            if pred[i] == 'I':
                res[i] = 'I'
    return res


def normalize(labels: List[str]) -> List[str]:
    """
    Normalize the labels of the corpus.
    Changing `begin`, `end` and `center` tags to `inside` tag.
    """
    res = []
    for label in labels:
        label = label[0]
        if label == 'B' or label == 'E' or label == 'C':
            res.append('I')
        else:
            res.append(label)
    return res


def get_idx_cues(y_predict: List[int]) -> List[List[int]]:
    "Gets the index of the cues in the sentence"
    res = []
    for pred in y_predict:
        cues = []
        for i, cue in enumerate(pred):
            if cue == 1:
                cues.append(i)
        res.append(cues)
    return res


def get_gold_cues(sentences: List[dict]) -> List[List[str]]:
    "Returns an array with the negation cues of the sentences in the corpus"
    res = []
    for sent in sentences:
        partial = []
        if 'cues' in sent:
            for cues in sent['cues']:
                partial.extend([c[1] for c in cues if c[2]])
        res.append(partial)
    return res


def get_gold_scope(sentences: List[dict]) -> List[List[str]]:
    "Returns an array with the negation scope of the sentences in the corpus"
    sent_labels = []
    for sent in sentences:
        labels = ['O' for key in sent if isinstance(key, int)]
        if 'scopes' in sent:
            for scope in sent['scopes']:
                for word in scope:
                    labels[word[1]] = 'I'
        sent_labels.append(labels)
    return sent_labels


def analize_coef(vect, clf):
    """
    Function to analize coefficients of selected classifier.
    """
    features = vect.inverse_transform(clf.coef_)[0]
    res = sorted(features, key=features.get)
    print('Most Positive features')
    for k in res[-10:]:
        print(k, features[k])
    print('\nMost Negative features')
    for k in res[:10]:
        print(k, features[k])

def get_cue_dep_path(graph, sent, cue_idx, word_idx):
    """
    Gets the shortest path between the actual word and the cue in the bidirectional dependency tree.
    Returns the length of the path and the path, otherwise returns null
    """
    try:
        path_list = nx.dijkstra_path(graph, word_idx, cue_idx)
        dep_path = ""
        # prev_node = word_idx
        for node in path_list:
            dep_path += sent[node]['dep'] + '/'
            # direction = graph[prev_node][node]['dir']
            # if direction == '/':
            #     dep_path += sent[node]['dep'] + '/'
            # else:
            #     dep_path += sent[prev_node]['dep'] + '\\'
            # prev_node = node
        return dep_path, make_discrete_distance(len(path_list))
    except:
        return 'null', 'null'


def get_head_cue_path(graph, sent, cue_idx, word_idx):
    """"
    Gets the path between the head of the cue and the current word in the actual directional dependecy graph
    Returns the path and its length if it exits, otherwise returns null
    """
    cue_head = sent[cue_idx]['head']
    try:
        path_list = nx.dijkstra_path(graph, cue_head, word_idx)
        dep_path = ""
        for node in path_list:
            dep_path += sent[node]['dep'] + '/'
        return dep_path, make_discrete_distance(len(path_list))
    except:
        return 'null', 'null'


def make_discrete_distance(dist):
    "Make discrete de distance of the selected method"
    if dist <= 3:
        return 'close'
    elif dist <= 7:
        return 'normal'
    elif dist > 7:
        return 'far'


def test_params(X, y, pos_label):
    "Test through grid search different parameters for Logistic Regression"
    parameters = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 15], 'max_iter': [100, 150, 200], 'penalty':['l1', 'l2']}
                    # 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag']}    #'newton-cg', 'lbfgs' and 'sag' only works for l2
    scorer = make_scorer(sklearn.metrics.f1_score, pos_label=pos_label)
    clf = GridSearchCV(sklearn.linear_model.LogisticRegression(), parameters, scoring=scorer, cv=10)
    clf.fit(X, y)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.5f (+/-%0.03f) for %r' %(mean, std*2, params))
    print('Best parameters found on development set: %s' % clf.best_params_)


def get_sent_feat(s: str) -> dict:
    "Get a sent from the user and extracts its features"
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


def reformat_prediction(features, cue_pred, scope_pred):
    """
    Reformats cue and scope predictions to express them in a
    human-readable format.
    """
    res = []
    for sent, c_p, s_p in zip(features, cue_pred, scope_pred):
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
