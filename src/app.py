from cgitb import text
from negation import get_cue_pred, get_scope_pred, load_clsf
import streamlit as st
from utils import get_idx_cues, get_sent_feat, reformat_prediction
from typing import List, Tuple

def change_color(prediction: List[Tuple[str, str]]):
    """
    Changes the color of the prediction to indicate the result of the classifiers.
    Color blue indicates out of scope, red indicates cue and orange indicates inside the scope.
    """
    texts = ['<p style="font-size: 20px; font-weight: bold">']
    for word, tag in prediction:
        if tag == 'C':
            texts.append(f'<span style="color:Red;">{word}</span>')
        elif tag == 'I':
            texts.append(f'<span style="color:Orange;">{word}</span>')
        elif tag == 'O':
            texts.append(f'<span style="color:Blue;">{word}</span>')
    texts.append('</p>')
    return ' '.join(texts)

st.title('Negation Detection in Spanish texts')

cue_clf, cue_vect, scope_clf, scope_vect = load_clsf()
sentence = st.text_input('Please enter a sentence: ')

if sentence != '':
    features = get_sent_feat(sentence)
    cue_pred = get_cue_pred(features, cue_clf, cue_vect)
    scope_pred = get_scope_pred(features, get_idx_cues(cue_pred), scope_clf, scope_vect)
    prediction = reformat_prediction(features, cue_pred, scope_pred)
    text_html = change_color(prediction)
    st.write('Result of the negation detection algorithm:')
    st.markdown(text_html, unsafe_allow_html=True)
    st.info('Color blue indicates the word is out of  of the negation scope, red indicates it\'s the negation cue and orange indicates it\'s inside the scope.')