from os import path, walk
from sys import path as p

from sklearn.externals import joblib
from cue_trainer import extract_cue_features
from scope_trainer import extract_scope_features
from parse_corpus import *
from evaluation import *
from utils import *
import spacy
import pickle

nlp = spacy.load('es_core_news_sm')

def read():
    """
    Starts reading the SFU_Review corpus. Returns a dictionary with the sentences in the corpus in a random order,
    with all its characteristics marked
    """
    pth = path.join(p[0], 'SFU_Review_SP_NEG')
    instances = []
    for (thisDir, _, filesHere) in walk(pth):
        thisDir = path.normpath(thisDir)
        for filename in filesHere:
            if filename.endswith('.xml'):
                file = path.join(thisDir, filename)
                for sent in read_file(file):
                    instances.append(sent)
    joblib.dump(instances, 'resources/sentences.pkl')
    # shuffle(instances)
    return instances


def start():
    create_model()
    # TEST: to get input sentences and detect the cues and its scope 
    # get_input()  


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


def get_cue_pred(instances, clf, vect):
    "Gets the predicted negatives cues"
    cue_feat = extract_cue_features(instances)
    dev_vect = vect.transform(cue_feat)
    predict = clf.predict(dev_vect)
    return make_labels_cue(instances, predict)


def get_scope_pred(instances, pred, clf, vect):
    "Gets the predicted negative scopes"
    dev_feat = extract_scope_features(instances, pred)
    X = vect.transform(dev_feat)
    y_predict = clf.predict(X)
    return make_labels_scope(instances, pred, y_predict)


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


def load_clsf():
    """Loads the classifier store in resources"""
    cue_clf = pickle.load(open(r"resources/cue_clf.pkl", 'rb'))
    cue_vect = pickle.load(open(r'resources/cue_vect.pkl', 'rb'))
    scope_clf = pickle.load(open(r'resources/scope_clf.pkl', 'rb'))
    scope_vect = pickle.load(open(r'resources/scope_vect.pkl', 'rb'))
    return cue_clf, cue_vect, scope_clf, scope_vect


def create_model():
    """
    Creates the model based on the corpus. 
    Prints the score of both classifier and the final test is optional
    """
    instances = joblib.load('resources/sentences.pkl')
    # lex = get_cue_lexicon(instances)
    
    train_size = int(len(instances)*0.8)
    train_set = instances[:train_size]
    dev_size = int(len(instances)*0.1) + train_size
    dev_set = instances[train_size:dev_size]
    test_set = instances[dev_size:]
    
    # Cue classifier
    clf, cue_vect = train_cue_learner(train_set)
    prediction = cue_model(clf, cue_vect, dev_set)

    # Scope classifier:
    labels = get_gold_cues(train_set)
    clsf, vect = train_scope_learner(train_set, labels)
    scope_model(clsf, vect, dev_set, prediction)
    # save_models(clf, cue_vect, clsf, vect)
    final_test_model(instances)


def save_models(cue_clsf, cue_vect, scope_clsf, scope_vect):
    "Save the trained models"
    pickle.dump(cue_clsf, open(r"resources/cue_clf.pkl", 'wb'))
    pickle.dump(cue_vect, open(r'resources/cue_vect.pkl', 'wb'))
    pickle.dump(scope_clsf, open(r'resources/scope_clf.pkl', 'wb'))
    pickle.dump(scope_vect, open(r'resources/scope_vect.pkl', 'wb'))


def cue_model(svm, vect, dev_set):
    "Tests the negation cue model"
    y_predict, y_true = test_cue_model(svm, vect, dev_set)
    # get_cue_errors(y_predict, y_true, dev_set)
    precision, recall, f1, accuracy = f1_evaluation(y_true, y_predict)
    print(metrics.flat_classification_report(y_true, y_predict))
    print("Precision: %.6f    Recall: %.6f  F1 Score: %.6f" % (precision, recall, f1))
    print('Accuracy: %.6f' %(accuracy))
    return get_idx_cues(y_predict)


def scope_model(clsf, vect, dev_set, prediction):
    "Tests the negation scope model"
    y_predict, y_true = test_scope_model(clsf, vect, dev_set, prediction)
    # get_scope_errors(y_predict, y_true, dev_set)
    print(metrics.flat_classification_report(y_true, y_predict))
    precision, recall, f1, accuracy = f1_evaluation(y_true, y_predict, pos_label="I")
    print("Precision: %.6f    Recall: %.6f  F1 Score: %.6f" %(precision, recall, f1))
    print('Accuracy: %.6f' %(accuracy))


start()