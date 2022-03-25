from negation import get_cue_pred, get_scope_pred, load_clsf
import streamlit as st
from utils import get_idx_cues, get_sent_feat, reformat_prediction


st.title('Negation Detection in Spanish texts')

cue_clf, cue_vect, scope_clf, scope_vect = load_clsf()
sentence = st.text_input('Please enter a sentence: ')
features = get_sent_feat(sentence)
cue_pred = get_cue_pred(features, cue_clf, cue_vect)
scope_pred = get_scope_pred(features, get_idx_cues(cue_pred), scope_clf, scope_vect)
prediction = reformat_prediction(features, cue_pred, scope_pred)
st.write(prediction)