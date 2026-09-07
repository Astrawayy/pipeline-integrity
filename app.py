# app.py — dark SaaS dashboard home
import streamlit as st
from common import setup_page, header, load_bundle, tile, bars_html, sidebar_nav

setup_page("Pipeline Integrity Predictor")
sidebar_nav()
header("Predicting wall thickness and inspection priority for oil and gas pipelines")

try:
    bundle=load_bundle(); ready=True
except Exception as e:
    ready=False
    st.error(f"Model file not found. Place pipeline_model_site.joblib next to app.py.  ({e})")

st.markdown('<div class="lede">A decision-support tool that estimates pipeline condition from '
            'operating parameters already on record, so inspection effort goes where it matters '
            'most. Rank a whole fleet, or assess a single line, in seconds.</div>',
            unsafe_allow_html=True)

st.markdown('<div class="sec">Model at a glance</div>', unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
c1.markdown(tile("g-cyan","Inspect accuracy","0.785","pipeline-level validation"),unsafe_allow_html=True)
c2.markdown(tile("g-green","Dangerous caught","89%","recall at a 9-year margin"),unsafe_allow_html=True)
c3.markdown(tile("g-violet","Pipelines","3,733","each measured ~30 times"),unsafe_allow_html=True)
c4.markdown(tile("g-amber","Parameters","27","across 5 categories"),unsafe_allow_html=True)

st.markdown('<div class="sec">What would you like to do?</div>', unsafe_allow_html=True)
c1,c2=st.columns(2, gap="medium")
c1.markdown('<div class="glass"><h4><span class="ico">🔧</span> Single pipeline</h4>'
            '<p>Enter the parameters of one line and get its remaining life, an inspect or leave '
            'decision, and a confidence gauge. Open <b>Single pipeline</b> in the sidebar.</p></div>',
            unsafe_allow_html=True)
c2.markdown('<div class="glass"><h4><span class="ico">📊</span> Whole fleet</h4>'
            '<p>Upload a CSV or Excel file and get every line back, ranked from most to least '
            'urgent, ready to download. Open <b>Fleet analysis</b>.</p></div>', unsafe_allow_html=True)

st.write("")
left,right=st.columns([1.4,1], gap="large")
with left:
    st.markdown('<div class="sec">What drives the prediction</div>', unsafe_allow_html=True)
    if ready and bundle.get("importances"):
        top=sorted(bundle["importances"].items(),key=lambda kv:-kv[1])[:8]
        st.markdown(bars_html(top), unsafe_allow_html=True)
with right:
    st.markdown('<div class="sec">How a prediction is built</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass"><p style="font-size:1.2rem;line-height:1.9;">'
        '<b style="color:#22D3EE;">1 &nbsp;Predict wall thickness</b><br>from the operating parameters.'
        '<br><br><b style="color:#22D3EE;">2 &nbsp;Derive corrosion rate</b><br>from nominal thickness and age.'
        '<br><br><b style="color:#22D3EE;">3 &nbsp;Compute remaining life</b><br>in years, with a range.'
        '<br><br><b style="color:#22D3EE;">4 &nbsp;Decide inspect or leave</b><br>erring toward flagging.'
        '</p></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Dago Engineering  ·  Binary inspection classifier plus thickness regressor, validated at pipeline level.")
