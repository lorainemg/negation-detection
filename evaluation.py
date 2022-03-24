from sklearn.model_selection import KFold

from utils import make_labels_cue, get_gold_cues, get_idx_cues
from cue_trainer import extract_cue_features, get_labels_cue, train_cue_learner
from scope_trainer import extract_scope_features, train_scope_learner
from pprint import pprint
import numpy as np

from utils import metrics, get_gold_scope, make_labels_scope


def test_cue_model(svm, vectorizer, dev_set):
    dev_feat = extract_cue_features(dev_set)
    dev_label = get_labels_cue(dev_set)
    dev_vect = vectorizer.transform(dev_feat)
    y_predict = svm.predict(dev_vect)
    # print(svm.score(dev_vect, dev_label))
    dev_y_predict = make_labels_cue(dev_set, y_predict)
    dev_y_true = make_labels_cue(dev_set, dev_label)
    # postprocessing(dev_y_predict, dev_set)
    return dev_y_predict, dev_y_true


def test_scope_model(clsf, vect, dev_set, prediction):
    dev_feat = extract_scope_features(dev_set, prediction)
    y_t = get_gold_scope(dev_set)
    X = vect.transform(dev_feat)
    y_predict = clsf.predict(X)
    y_p = make_labels_scope(dev_set, prediction, y_predict)
    return y_p, y_t


def f1_evaluation(y_true, y_predict, pos_label=1):
    precision = metrics.flat_precision_score(y_true, y_predict, pos_label=pos_label)
    recall = metrics.flat_recall_score(y_true, y_predict, pos_label=pos_label)
    f1 = metrics.flat_f1_score(y_true, y_predict, pos_label=pos_label)
    accuracy = metrics.flat_accuracy_score(y_true, y_predict)
    return precision, recall, f1, accuracy


def cross_validation(sentences):
    kf = KFold(n_splits=30)
    score_cue = [[], [], [], []]
    score_scope = [[], [], [], []]
    scr_scope_gold = [[], [], [], []]
    sent = np.array(sentences)
    for train, test in kf.split(sent):
        train_set = sent[train]
        test_set = sent[test]
        svm, vect = train_cue_learner(train_set)
        y_p, y_t = test_cue_model(svm, vect, test_set)
        evaluation = f1_evaluation(y_t, y_p)
        score_cue[0].append(evaluation[0])            # prec
        score_cue[1].append(evaluation[1])            # recall
        score_cue[2].append(evaluation[2])            # f1
        score_cue[3].append(evaluation[3])            # accuracy

        labels = get_gold_cues(train_set)
        clsf, vect = train_scope_learner(train_set, labels)
        prediction = get_idx_cues(y_p)
        y_predict, y_true = test_scope_model(clsf, vect, test_set, prediction)
        evaluation = f1_evaluation(y_true, y_predict, pos_label='I')
        score_scope[0].append(evaluation[0])            # prec
        score_scope[1].append(evaluation[1])            # recall
        score_scope[2].append(evaluation[2])            # f1
        score_scope[3].append(evaluation[3])            # accuracy

        labels = get_gold_cues(train_set)
        clsf, vect = train_scope_learner(train_set, labels)
        prediction = get_gold_cues(test_set)
        y_predict, y_true = test_scope_model(clsf, vect, test_set, prediction)
        evaluation = f1_evaluation(y_true, y_predict, pos_label='I')
        scr_scope_gold[0].append(evaluation[0])            # prec
        scr_scope_gold[1].append(evaluation[1])            # recall
        scr_scope_gold[2].append(evaluation[2])            # f1
        scr_scope_gold[3].append(evaluation[3])            # accuracy
    return score_cue, score_scope, scr_scope_gold


def get_cue_errors(y_predict, y_true, sent):
    i, count = 0, 0
    for predict, gold in zip(y_predict, y_true):
        if predict != gold:
            print_cue(sent[i], predict, gold)
            count += 1
        i += 1
    print(count)


def get_scope_errors(y_predict, y_true, sent):
    i, count = 0, 0
    for predict, gold in zip(y_predict, y_true):
        if predict != gold:
            print_scope(sent[i], predict, gold)
            count += 1
        i += 1
    print(count)


def print_cue(sent, predict, gold):
    print(sent['tokens'])
    pprint(sent['cues'])
    pprint(sent['mw_cues'])
    print(predict)
    print(gold)
    print()
    return


def print_scope(sent, predict, gold):
    print(sent['tokens'])
    print(sent['scopes'])
    print(predict)
    print(gold)
    print()
    return


def final_test_model(instances):
    """Prints the cross validation score of the cue and the scope classifier"""
    scores_cue, scores_scope, scr_gold_scp = cross_validation(instances)
    prec_cue, recall_cue, f1_cue , acc_cue = scores_cue
    print('-----------------------------------CUE----------------------------------------------')
    print('Precision:', prec_cue, '\nAverage precision:', sum(prec_cue)/len(prec_cue))
    print('Recall:', recall_cue,  '\nAverage recall:', sum(recall_cue)/len(recall_cue))
    print('F1:', f1_cue,          '\nAverage f1:', sum(f1_cue)/len(f1_cue))
    print('Accuracy:', acc_cue,   '\nAverage accuracy:', sum(acc_cue)/len(acc_cue))
    prec_scope, recall_scope, f1_scope, acc_scope = scores_scope
    print('\n-----------------------------------SCOPE----------------------------------------------')
    print('Precision:', prec_scope, '\nAverage precision:', sum(prec_scope)/len(prec_scope))
    print('Recall:', recall_scope,  '\nAverage recall:', sum(recall_scope)/len(recall_scope))
    print('F1:', f1_scope,          '\nAverage f1:', sum(f1_scope)/len(f1_scope))
    print('Accuracy:', acc_scope,   '\nAverage accuracy:', sum(acc_scope)/len(acc_scope))
    prec_scope, recall_scope, f1_scope, acc_scope = scr_gold_scp
    print('\n-----------------------------------SCOPE GOLD----------------------------------------------')
    print('Precision Gold:', prec_scope, '\nAverage precision gold:', sum(prec_scope)/len(prec_scope))
    print('Recall Gold:', recall_scope,  '\nAverage recall gold:', sum(recall_scope)/len(recall_scope))
    print('F1 Gold:', f1_scope,          '\nAverage f1 gold:', sum(f1_scope)/len(f1_scope))
    print('Accuracy Gold:', acc_scope,   '\nAverage accuracy gold:', sum(acc_scope)/len(acc_scope))