import streamlit as st
from common import (setup_page, header, load_bundle, predict_one,
                    key_fields, render_field, result_cards, GROUPS, sidebar_nav)

setup_page("Single pipeline")
sidebar_nav()
header("Analyse one pipeline and get its remaining life and inspection decision")

bundle=load_bundle(); KEY=key_fields(bundle)

st.markdown('<div class="lede">Every field starts at a typical value. Change the ones you know and '
            'leave the rest. The more you fill in, the tighter the estimate.</div>',
            unsafe_allow_html=True)

left,right=st.columns([1,1], gap="large")

with left:
    st.markdown('<div class="sec">Key parameters</div>', unsafe_allow_html=True)
    ui={"Material":"API5LGRB"}
    with st.form("pipe_form"):
        cols=st.columns(2)
        for i,f in enumerate(KEY):
            v=render_field(f,cols[i%2],bundle)
            if v is not None: ui[f]=v
        with st.expander("All other parameters (optional)"):
            for group,fields in GROUPS.items():
                rest=[f for f in fields if f not in KEY]
                if not rest: continue
                st.markdown(f"**{group}**")
                gcols=st.columns(2)
                for i,f in enumerate(rest):
                    v=render_field(f,gcols[i%2],bundle)
                    if v is not None: ui[f]=v
        st.write("")
        submitted=st.form_submit_button("Estimate condition", type="primary",
                                        use_container_width=True)
    if submitted:
        st.session_state["last"]=predict_one(ui,bundle)

with right:
    st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)
    if st.session_state.get("last"):
        result_cards(st.session_state["last"])
    else:
        st.markdown('<div class="glass" style="height:200px;display:flex;align-items:center;">'
                    '<p>Fill in the parameters and press <b>Estimate condition</b>. The decision, '
                    'a confidence gauge and the remaining-life range appear here.</p></div>',
                    unsafe_allow_html=True)
