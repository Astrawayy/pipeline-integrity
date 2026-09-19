import pandas as pd
import streamlit as st
from common import setup_page, header, load_bundle, predict_batch, tile, sidebar_nav

setup_page("Fleet analysis")
sidebar_nav()
header("Upload a fleet and get every line ranked by inspection priority")

bundle = load_bundle()

# Pandas Styler refuses to colour more than ~262k cells, so we only style a preview.
PREVIEW_ROWS = 300

st.markdown('<div class="lede">Upload one pipeline per row, in CSV or Excel. Include any subset of '
            'the parameters; anything missing is filled automatically. Results come back sorted with '
            'the most urgent lines first.</div>', unsafe_allow_html=True)

st.markdown('<div class="sec">Upload your file</div>', unsafe_allow_html=True)
up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
with st.expander("Which column names does it recognise?"):
    st.write(", ".join(bundle["features"]))

if up is not None:
    try:
        if up.name.lower().endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(up, sheet_name=None)
            data = (sheets[st.selectbox("Which sheet?", list(sheets.keys()))]
                    if len(sheets) > 1 else list(sheets.values())[0])
        else:
            data = pd.read_csv(up)

        recognised = [c for c in data.columns if c in bundle["features"]]
        if not recognised:
            st.warning("None of the column names match the model's parameters.")
        else:
            with st.spinner(f"Scoring {len(data):,} pipelines..."):
                result = predict_batch(data, bundle).sort_values("_sort").drop(columns="_sort")

            n = len(result)
            ni = int((result["Decision"] == "Inspect").sum())

            st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(tile("g-violet", "Pipelines", f"{n:,}",
                             f"{len(recognised)}/{len(bundle['features'])} params"), unsafe_allow_html=True)
            c2.markdown(tile("g-red", "To inspect", f"{ni:,}", "flagged"), unsafe_allow_html=True)
            c3.markdown(tile("g-green", "No action", f"{n-ni:,}", "healthy"), unsafe_allow_html=True)
            c4.markdown(tile("g-amber", "Share flagged", f"{ni/max(n,1)*100:.0f}%", "of fleet"),
                        unsafe_allow_html=True)

            # ── Risk class breakdown, when the column is available ───────────
            if "Risk class" in result.columns:
                st.markdown('<div class="sec">Risk classes (API 570)</div>', unsafe_allow_html=True)
                counts = result["Risk class"].value_counts()
                order = [("Critical", "g-red"), ("Monitor", "g-amber"),
                         ("Normal", "g-violet"), ("Healthy", "g-green")]
                cols = st.columns(4)
                for col, (lab, style) in zip(cols, order):
                    v = int(counts.get(lab, 0))
                    col.markdown(tile(style, lab, f"{v:,}", f"{v/max(n,1)*100:.1f}% of fleet"),
                                 unsafe_allow_html=True)

            # ── Ranked results: style only the preview, download holds it all ─
            st.markdown('<div class="sec">Ranked results</div>', unsafe_allow_html=True)
            preview = result.head(PREVIEW_ROWS)
            if n > PREVIEW_ROWS:
                st.caption(f"Showing the {PREVIEW_ROWS} most urgent lines out of {n:,}. "
                           "The download below contains every pipeline.")

            def hl(row):
                c = "#3a1f27" if row["Decision"] == "Inspect" else "#163029"
                return [f"background-color: {c}; color:#EAF2FA"] * len(row)

            try:
                st.dataframe(preview.style.apply(hl, axis=1), use_container_width=True, height=560)
            except Exception:
                # very wide files can still exceed the styler budget: fall back to plain
                st.dataframe(preview, use_container_width=True, height=560)

            st.download_button("Download full results as CSV",
                               result.to_csv(index=False).encode("utf-8"),
                               "pipeline_predictions.csv", "text/csv",
                               use_container_width=True)
    except Exception as e:
        st.error(f"Could not read that file: {e}")
else:
    st.markdown('<div class="glass"><p>No file yet. Export a few rows from your dataset to try it: in '
                'the notebook, <code>df[feat].head(20).to_csv("test.csv", index=False)</code>, then '
                'upload <b>test.csv</b> here.</p></div>', unsafe_allow_html=True)
