"""Module to parse the corpus"""
from nltk import data
from xml.etree.ElementTree import ElementTree
import spacy
import networkx as nx

nlp = spacy.load('es_core_news_sm')

def read_file(filename):
    """
    Reads the file, its supossed to have the format of the sfu corpus.
    Returns a dictionary representing the sentences in the corpus with all its data stored
    """
    sfu_file = data.find(filename)
    sfu = ElementTree().parse(sfu_file)
    sentences = []
    for sent in sfu.getchildren():
        sentence = {}
        cues = []
        tokens = []
        mw_cues = []
        scopes = []
        i = 0
        if len(sent.items()) > 0:
            sentence['neg'] = True
        else:
            sentence['neg'] = False
        sentence['file'] = filename
        for word in sent.getchildren():
            w = {}
            scp = []
            c = []
            mwc = []
            if word.tag == 'neg_structure':
                i = get_structure(scp, c, mwc, word, sentence, i, False, tokens)
                if len(scp) > 0:
                    scopes.append(scp)
                if len(c) > 0:
                    cues.append(c)
                if len(mwc) > 0:
                    mw_cues.append(mwc)
            elif word.tag == 'scope':
                i = get_scope(scp, c, mwc, word, sentence, i, False, tokens, False)
            else:
                w = dict(word.items())
                if w.get('elliptic') == 'yes':
                    continue
                key = 'wd'
                if not w.__contains__(key):
                    key = 'lem'
                tokens.append(w[key])
                sentence[i] = w
                i += 1
        sentence['cues'] = cues
        sentence['mw_cues'] = mw_cues
        sentence['scopes'] = scopes
        sentence['tokens'] = tokens
        sentence['length'] = i
        get_dependency_graph(tokens, sentence)
        sentences.append(sentence)
    return sentences


def get_dependency_graph(tokens, sentence):
    """
    Makes all the analysis of the sentence according to spacy
    :param tokens:
    :param sentence:
    :return:
    """
    s = ' '.join(tokens)
    doc = nlp(s)
    edges = []
    graph = nx.DiGraph()
    digraph = nx.DiGraph()
    for token in doc:
        for child in token.children:
            digraph.add_edge(token.i, child.i)
            graph.add_edge(token.i, child.i, {'dir': '/'})
            graph.add_edge(child.i, token.i, {'dir': '\\'})
            # edges.append((token.i, child.i))
        if token.i in sentence:
            sentence[token.i]['head'] = token.head.i
            sentence[token.i]['dep'] = token.dep_
            sentence[token.i]['pos'] = token.pos_
            sentence[token.i]['lemma'] = token.lemma_
    sentence['undir-graph'] = graph #nx.Graph(edges)
    sentence['dir-graph'] = digraph #nx.DiGraph(edges)


def get_structure(scp, cues, mw_cues, word, sentence, i, neg, tokens):
    "Parses the negative structure of the sentence"
    scope = word.getchildren()[0]
    items = word.items()
    neg_val = not any(i == ('value', 'noneg') for i in items)
    for component in word.getchildren():
        if component.tag == 'scope':
            i = get_scope(scp, cues, mw_cues, component, sentence, i, neg, tokens, neg_val)
        elif component.tag == 'negexp':
            i = get_neg(component, sentence, cues, mw_cues, scp, i, tokens, neg_val)
        elif component.tag == 'event':
            i = skip_event(component, sentence, scp, i, cues, mw_cues, neg, tokens, neg_val)
        elif component.tag == 'neg_structure':
            i = get_structure(scp, cues, mw_cues, component, sentence, i, neg, tokens)
        else:
            w = dict(component.items())
            key = 'wd'
            if not w.__contains__(key):
                key = 'lem'
            tokens.append(w[key])
            sentence[i] = w
            i += 1
    return i


def get_scope(scp, cues, mw_cues, scope, sentence, i, neg, tokens, neg_val):
    "Gets the scope of the sentence, getting its cues and scopes"
    for component in scope.getchildren():
        if component.tag == 'neg_structure':
            i = get_structure(scp, cues, mw_cues, component, sentence, i, neg, tokens)
        elif component.tag == 'scope':
            i = get_scope(scp, cues, mw_cues, component, sentence, i, neg, tokens, neg_val)
        elif component.tag == 'negexp':
            i = get_neg(component, sentence, cues, mw_cues, scp, i, tokens, neg_val)
        elif component.tag == 'event' or component.tag == 'infinitiu':
            i = skip_event(component, sentence, scp, i, cues, mw_cues, neg, tokens, neg_val)
        elif component.tag == 'word':
            continue
        else:
            w = dict(component.items())
            sentence[i] = w
            tokens.append(w['wd'])
            scp.append((w['wd'], i))
            i += 1
    return i


def get_neg(component, sentence, cues, mw_cues, scp, i, tokens, neg_val):
    "Get the negative cues, including simple and multiwords cues"
    for word in component.getchildren():
        if word.tag == 'negexp':
            i = get_neg(word, sentence, cues, mw_cues, scp, i, tokens, neg_val)
        elif word.tag == 'event':
            i = skip_event(word, sentence, scp, i, cues, mw_cues, True, tokens, neg_val)
        elif word.tag == 'S' or word.tag == 'word':
            continue
        else:
            w = dict(word.items())
            if len(w) == 0:
                continue
            sentence[i] = w
            key = 'wd'
            if not w.__contains__('wd'):
                key = 'lem'
            tokens.append(w[key])
            scp.append((w[key], i))
            cues.append((w[key], i, neg_val))
            i += 1
    return i


def skip_event(component, sentence, scp, i, cues, mw_cues, neg, tokens, neg_val):
    "Skips the informative event in the corpus, because we are not going to parsed it"
    for word in component.getchildren():
        if word.tag == 'neg_structure':
            i = get_structure(scp, cues, mw_cues, word, sentence, i, neg, tokens)
        elif word.tag == 'scope':
            i = get_scope(scp, cues, mw_cues, word, sentence, i, neg, tokens, neg_val)
        elif word.tag == 'negexp':
            i = get_neg(word, sentence, cues, mw_cues, scp, i, tokens, neg_val)
        elif word.tag == 'event':
            i = skip_event(word, sentence, scp, i, cues, mw_cues, neg, tokens, neg_val)
        elif word.tag == 'word' or word.tag == 'S':
            continue
        else:
            w = dict(word.items())
            sentence[i] = w
            scp.append((w['wd'], i))
            tokens.append(w['wd'])
            if neg:
                cues.append((w['wd'], i, neg_val))
            i += 1
    return i
