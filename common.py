# common.py — shared helpers: model, prediction, dark SaaS styling, header, cards
import os, base64
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BUNDLE_PATH = "pipeline_model_site.joblib"
LOGO_CANDIDATES = ["logo_dago.png", "logo.png", "dago_logo.png"]
N_KEY_FIELDS = 8

# ── Dark palette ─────────────────────────────────────────────────────────────
BG="#0B1622"; BG2="#0F1E2E"; PANEL="#132638"; PANEL2="#17304A"
BORDER="#21405E"; INK="#EAF2FA"; SOFT="#9DB4C9"; MUTE="#6E869E"
CYAN="#22D3EE"; BLUE="#38BDF8"; TEAL="#2DD4BF"; GREEN="#4ADE80"
AMBER="#FCD34D"; RED="#FB7185"; VIOLET="#A78BFA"

# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)

def predict_one(user_inputs, b):
    cfg=b["config"]; cap=cfg["RL_CAP"]
    row={c:user_inputs.get(c,b["medians"][c]) for c in b["features"]}
    n_given=sum(1 for c in b["features"] if user_inputs.get(c) is not None)
    coverage=n_given/len(b["features"])
    X=pd.DataFrame([row])[b["features"]]
    proba=b["classifier"].predict_proba(X)[0]
    classes=list(b["classifier"].named_steps["rf"].classes_)
    decision=classes[int(np.argmax(proba))]
    confidence=0.5+(float(np.max(proba))-0.5)*coverage
    pre=b["regressor"].named_steps["pre"]; forest=b["regressor"].named_steps["rf"]
    Xt=pre.transform(X)
    tree_thk=np.array([t.predict(Xt)[0] for t in forest.estimators_])
    nominal=float(row[b["nominal_col"]]); age=max(float(row[b["age_col"]]),1e-6)
    tcrit=cfg["T_CRIT_FRAC"]*nominal
    cr=np.maximum((nominal-tree_thk)/age,1e-6)
    tree_rl=np.clip((tree_thk-tcrit)/cr,0,cap)
    rl_median=float(np.median(tree_rl))
    rl_low,rl_high=float(np.percentile(tree_rl,10)),float(np.percentile(tree_rl,90))
    capped=rl_median>=cap
    if rl_median<cfg["SAFETY_MARGIN"]: decision="Inspect"
    return {"decision":decision,"confidence":confidence,"rl_median":rl_median,
            "rl_display":f"{cap:.0f}+" if capped else f"{rl_median:.1f}",
            "interval_low":rl_low,"interval_high":min(rl_high,cap),
            "capped":capped,"coverage":coverage,
            "fields_provided":n_given,"fields_total":len(b["features"]),
            "tree_rl":tree_rl}

def predict_batch(dframe,b):
    out=[predict_one({c:r[c] for c in b["features"]
                      if c in dframe.columns and pd.notna(r[c])},b)
         for _,r in dframe.iterrows()]
    res=pd.DataFrame(out); keep=dframe.copy()
    keep.insert(0,"Decision",res["decision"].values)
    keep.insert(1,"Remaining life (yr)",res["rl_display"].values)
    keep.insert(2,"Range",[f"{a:.0f} to {b_:.0f}" for a,b_ in
                           zip(res["interval_low"],res["interval_high"])])
    keep.insert(3,"Confidence",(res["confidence"]*100).round(0).astype(int).astype(str)+"%")
    keep["_sort"]=res["rl_median"].values
    return keep

# ── Field metadata ───────────────────────────────────────────────────────────
NUM_STEP={"Nominal Pipe Size":1.0,"Thickness Nominal (mm)":0.1,"Year Built":1.0,
    "Design Pressure (psi)":5.0,"Operating Pressure (psi)":5.0,"Operating Temperature (F)":5.0,
    "Flow Rate (BFPD)":100.0,"Service Age":1.0,"Water Cut":1.0,"HCA":1.0,"PoF":1.0,
    "Soil pH":0.1,"CO2":1.0,"H2S":1.0,"LeakCount":1.0}
SHORT={"Thickness Nominal (mm)":"Nominal thickness (mm)",
    "Fluid Representative (Heavy, Light, Gas, Condensate)":"Fluid type (detailed)",
    "Coating (Painting, Coating, Known/Unknown)":"Coating condition",
    "Operating Temperature (F)":"Operating temp. (F)","CP (Y/N)":"Cathodic protection",
    "Insulation (Y/N)":"Insulation","PoF":"Probability of failure (1-5)",
    "CoF":"Consequence of failure (A-E)","HCA":"High consequence area","LeakCount":"Past leak count"}
GROUPS={"Geometry":["Nominal Pipe Size","Thickness Nominal (mm)","Pipe Type","Pipe Position","Joint Type"],
    "Operation":["Year Built","Service Age","Design Pressure (psi)","Operating Pressure (psi)",
                 "Operating Temperature (F)","Flow Rate (BFPD)"],
    "Fluid and chemistry":["Service Fluid Type","Fluid Representative (Heavy, Light, Gas, Condensate)",
                           "Water Cut","CO2","H2S"],
    "Environment and protection":["Soil pH","Soil Resistivity","Coating (Painting, Coating, Known/Unknown)",
                                  "CP (Y/N)","Insulation (Y/N)","Sub Area"],
    "Risk indicators":["HCA","PoF","CoF","LeakCount"]}

def key_fields(b):
    imp=b.get("importances",{})
    if imp:
        ranked=[f for f,_ in sorted(imp.items(),key=lambda kv:-kv[1]) if f in b["features"]]
    else:
        ranked=["Operating Pressure (psi)","Flow Rate (BFPD)","Thickness Nominal (mm)","Year Built",
                "Service Age","Water Cut","Nominal Pipe Size","Sub Area"]
    return ranked[:N_KEY_FIELDS]

def label_of(f): return SHORT.get(f,f)

def render_field(f,col,b):
    NUM=b["num_ranges"]; CAT=b["cat_options"]
    if f in NUM:
        lo,hi,med=NUM[f]["min"],NUM[f]["max"],NUM[f]["median"]
        return col.number_input(label_of(f),min_value=float(lo),max_value=float(hi),
                                value=float(med),step=NUM_STEP.get(f,1.0))
    if f in CAT:
        opts=CAT[f]; med=str(b["medians"][f])
        return col.selectbox(label_of(f),opts,index=opts.index(med) if med in opts else 0)
    return None

# ── Logo + banner ────────────────────────────────────────────────────────────
def _logo_b64():
    for name in LOGO_CANDIDATES:
        if os.path.exists(name):
            with open(name,"rb") as f: return base64.b64encode(f.read()).decode()
    return None

def setup_page(title="Pipeline Integrity"):
    st.set_page_config(page_title=title,page_icon="🛢️",layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(f"""
    <style>
      .stApp {{ background:
        radial-gradient(1200px 600px at 15% -10%, #16324e 0%, rgba(22,50,78,0) 55%),
        radial-gradient(1000px 500px at 100% 0%, #123047 0%, rgba(18,48,71,0) 50%),
        {BG}; }}
      .block-container {{ padding:1.2rem 2.6rem 3.5rem 2.6rem !important; max-width:100% !important; }}
      #MainMenu, footer, header[data-testid="stHeader"] {{ display:none; }}
      .stApp, .stApp p, .stApp span, .stApp div, .stApp label {{ color:{INK}; }}

      section[data-testid="stSidebar"] {{ background:{BG2}; border-right:1px solid {BORDER};
          min-width:300px !important; }}
      /* hide Streamlit's default page list, we render our own */
      [data-testid="stSidebarNav"] {{ display:none !important; }}
      section[data-testid="stSidebar"] .block-container {{ padding-top:1.6rem; }}

      /* custom nav buttons */
      [data-testid="stPageLink"] a, a[data-testid="stPageLink-NavLink"] {{
          display:flex !important; align-items:center; justify-content:center;
          background:linear-gradient(160deg, rgba(23,48,74,.9), rgba(15,30,46,.75)) !important;
          border:1px solid {BORDER} !important; border-radius:16px !important;
          padding:1.25rem 1rem !important; margin-bottom:.85rem !important;
          transition:all .18s ease; text-decoration:none !important; }}
      [data-testid="stPageLink"] a p, a[data-testid="stPageLink-NavLink"] p {{
          font-size:1.35rem !important; font-weight:800 !important; color:{SOFT} !important;
          margin:0 !important; letter-spacing:.3px; }}
      [data-testid="stPageLink"] a:hover, a[data-testid="stPageLink-NavLink"]:hover {{
          border-color:{CYAN} !important; transform:translateY(-2px);
          box-shadow:0 12px 28px rgba(34,211,238,.32) !important; }}
      [data-testid="stPageLink"] a:hover p, a[data-testid="stPageLink-NavLink"]:hover p {{
          color:{INK} !important; }}
      .sb-logo {{ background:#fff; border-radius:16px; padding:14px 16px; margin:0 .3rem 1.4rem .3rem;
          box-shadow:0 8px 22px rgba(0,0,0,.4); text-align:center; }}
      .sb-logo img {{ width:100%; max-width:240px; display:block; margin:0 auto; }}
      .navtitle {{ font-size:1.35rem; letter-spacing:2px; text-transform:uppercase;
          color:{CYAN}; font-weight:800; margin:0 0 1.1rem 0; text-align:center;
          text-shadow:0 0 16px rgba(34,211,238,.5); }}

      .lede {{ color:{SOFT}; font-size:1.55rem; line-height:1.55; max-width:1150px; margin:.6rem 0 0; }}
      .sec {{ font-size:1.3rem; font-weight:800; letter-spacing:1.2px; text-transform:uppercase;
              color:{CYAN}; margin:2rem 0 1rem; display:flex; align-items:center; gap:.7rem;
              text-shadow:0 0 18px rgba(34,211,238,.45); }}
      .sec::before {{ content:""; width:34px; height:3px; border-radius:2px;
                      background:linear-gradient(90deg,{CYAN},transparent); display:inline-block; }}

      /* glass hero */
      .hero {{ position:relative; border-radius:22px; overflow:hidden; margin-bottom:.2rem;
               border:1px solid {BORDER};
               background:linear-gradient(120deg, rgba(23,48,74,.9), rgba(15,30,46,.75));
               box-shadow:0 20px 60px rgba(0,0,0,.45); }}
      .hero-bg {{ position:absolute; inset:0; opacity:.5; }}
      .hero-ov {{ position:relative; display:grid; grid-template-columns:1fr auto 1fr;
                  align-items:center; gap:1.4rem; padding:2.2rem 2.4rem; }}
      .hero-logo {{ height:150px; background:#fff; border-radius:20px; padding:14px 22px;
                    box-shadow:0 10px 30px rgba(0,0,0,.45); justify-self:start; }}
      .hero-txt {{ text-align:center; align-self:center; padding-bottom:1.4rem; }}
      .hero-txt h1 {{ margin:0; font-size:3.1rem; font-weight:900; letter-spacing:-1px; color:#fff;
                      text-shadow:0 0 28px rgba(34,211,238,.45), 0 2px 12px rgba(0,0,0,.5); }}
      .hero-txt p {{ margin:.6rem 0 0; color:#DCEBF7; font-size:1.5rem; }}
      .hero-badge {{ justify-self:end; text-align:right; }}
      .hero-badge .b1 {{ font-size:.85rem; letter-spacing:1.6px; text-transform:uppercase; color:{SOFT}; }}
      .hero-badge .b2 {{ font-size:1.35rem; font-weight:800; color:{TEAL};
                         text-shadow:0 0 16px rgba(45,212,191,.6); }}
      .hero-badge .b3 {{ font-size:1.4rem; font-weight:900; color:#22C55E; margin-top:.15rem;
                         text-shadow:0 0 18px rgba(34,197,94,.85), 0 0 4px rgba(34,197,94,.5); }}

      /* glass cards */
      .glass {{ border-radius:18px; padding:1.35rem 1.5rem; height:100%;
                background:linear-gradient(160deg, rgba(23,44,66,.7), rgba(16,32,48,.55));
                border:1px solid {BORDER}; backdrop-filter:blur(8px);
                box-shadow:0 10px 30px rgba(0,0,0,.28); }}
      .glass h4 {{ margin:0 0 .7rem; font-size:1.7rem; font-weight:800; color:{INK};
                   display:flex; align-items:center; gap:.6rem; }}
      .glass p {{ margin:0; color:{SOFT}; font-size:1.32rem; line-height:1.55; }}
      .ico {{ font-size:1.8rem; filter:drop-shadow(0 0 12px rgba(34,211,238,.7)); }}

      /* stat tiles with glow */
      .tile {{ border-radius:18px; padding:1.4rem 1.5rem; height:100%; position:relative; overflow:hidden;
               border:1px solid {BORDER};
               background:linear-gradient(160deg, rgba(23,44,66,.75), rgba(16,32,48,.6)); }}
      .tile::after {{ content:""; position:absolute; top:-40%; right:-20%; width:170px; height:170px;
                      border-radius:50%; filter:blur(30px); opacity:.5; }}
      .tile .k {{ font-size:1.2rem; letter-spacing:1.3px; text-transform:uppercase; color:{SOFT};
                  font-weight:700; }}
      .tile .v {{ font-size:3.6rem; font-weight:900; line-height:1.02; margin-top:.35rem; }}
      .tile .s {{ font-size:1.28rem; color:{SOFT}; margin-top:.4rem; }}
      .g-cyan::after {{ background:{CYAN}; }} .g-cyan .v {{ color:{CYAN}; text-shadow:0 0 22px rgba(34,211,238,.65); }}
      .g-green::after {{ background:{GREEN}; }} .g-green .v {{ color:{GREEN}; text-shadow:0 0 22px rgba(74,222,128,.6); }}
      .g-violet::after {{ background:{VIOLET}; }} .g-violet .v {{ color:{VIOLET}; text-shadow:0 0 22px rgba(167,139,250,.6); }}
      .g-amber::after {{ background:{AMBER}; }} .g-amber .v {{ color:{AMBER}; text-shadow:0 0 22px rgba(252,211,77,.6); }}
      .g-red::after {{ background:{RED}; }} .g-red .v {{ color:{RED}; text-shadow:0 0 22px rgba(251,113,133,.6); }}

      /* inputs */
      div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {{
          font-size:1.45rem !important; color:{INK} !important; font-weight:700 !important; }}
      div[data-testid="stNumberInput"] input, div[data-baseweb="select"] div {{
          font-size:1.3rem !important; }}
      .stExpander summary, .stExpander p {{ font-size:1.25rem !important; }}
      div[data-testid="stNumberInput"] input {{ background:{PANEL} !important; color:{INK} !important;
          border:1px solid {BORDER} !important; }}
      div[data-baseweb="select"]>div {{ background:{PANEL} !important; border-color:{BORDER} !important; }}
      div[data-testid="stForm"] {{ border:1px solid {BORDER}; border-radius:20px; padding:1.2rem 1.5rem;
          background:linear-gradient(160deg, rgba(19,38,56,.6), rgba(13,26,40,.5)); }}
      .stButton>button, .stDownloadButton>button, div[data-testid="stFormSubmitButton"]>button {{
          background:linear-gradient(90deg,#0EA5C4,#2A7FC0) !important; color:#EAF7FF !important;
          border:none !important; font-weight:800 !important; border-radius:12px !important;
          box-shadow:0 4px 14px rgba(14,165,196,.2) !important;
          font-size:1.4rem !important; padding:.95rem 1.4rem !important; }}
      .stExpander {{ border:1px solid {BORDER} !important; border-radius:14px !important;
          background:rgba(19,38,56,.4) !important; }}
      .stProgress > div > div > div {{ background:linear-gradient(90deg,{CYAN},{TEAL}) !important; }}
      [data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; }}
      hr {{ border-color:{BORDER} !important; }}
    </style>""", unsafe_allow_html=True)

# subtle pipe pattern behind the hero
HERO_BG=('<svg class="hero-bg" viewBox="0 0 1600 220" preserveAspectRatio="xMidYMid slice" '
  'xmlns="http://www.w3.org/2000/svg">'
  '<defs><linearGradient id="pp" x1="0" y1="0" x2="0" y2="1">'
  '<stop offset="0" stop-color="#2b5f8a"/><stop offset="1" stop-color="#14324c"/></linearGradient>'
  '<radialGradient id="gl" cx="0.5" cy="0.5" r="0.5">'
  '<stop offset="0" stop-color="#38BDF8" stop-opacity="0.55"/>'
  '<stop offset="1" stop-color="#38BDF8" stop-opacity="0"/></radialGradient></defs>'
  '<circle cx="1300" cy="30" r="180" fill="url(#gl)"/>'
  '<g opacity="0.5"><rect x="0" y="150" width="1600" height="26" rx="13" fill="url(#pp)"/>'
  '<rect x="300" y="142" width="16" height="42" rx="4" fill="#0f2438"/>'
  '<rect x="820" y="142" width="16" height="42" rx="4" fill="#0f2438"/>'
  '<rect x="1300" y="142" width="16" height="42" rx="4" fill="#0f2438"/>'
  '<circle cx="560" cy="163" r="15" fill="#0B1622" stroke="#38BDF8" stroke-width="2"/>'
  '<circle cx="1080" cy="163" r="15" fill="#0B1622" stroke="#38BDF8" stroke-width="2"/></g></svg>')

def header(subtitle="Remaining life and inspection priority", badge="Dago Engineering"):
    logo=_logo_b64()
    logo_html=(f'<img src="data:image/png;base64,{logo}" class="hero-logo"/>' if logo else "")
    html=('<div class="hero">'+HERO_BG+'<div class="hero-ov">'+logo_html+
          '<div class="hero-txt"><h1>Pipeline Integrity Predictor</h1><p>'+subtitle+'</p></div>'
          '<div class="hero-badge"><div class="b1">Powered by</div>'
          '<div class="b2">Machine Learning</div>'
          '<div class="b3">Random Forest</div></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)

def tile(kind,label,value,sub=""):
    return (f'<div class="tile {kind}"><div class="k">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')

# ── Gauge + interval charts (matplotlib, dark) ───────────────────────────────
def _fig_to_b64(fig):
    import io
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=130,transparent=True,bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def confidence_gauge(conf):
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(3.8,2.2),subplot_kw={"projection":"polar"})
    ax.set_theta_offset(np.pi); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    col = GREEN if conf>=0.75 else (AMBER if conf>=0.6 else RED)
    ax.barh(1,np.pi,height=0.55,left=0,color="#1b3350")
    ax.barh(1,np.pi*conf,height=0.55,left=0,color=col)
    ax.set_axis_off()
    ax.text(np.pi/2,-0.35,f"{conf*100:.0f}%",ha="center",va="center",
            fontsize=40,fontweight="bold",color="#EAF2FA")
    return _fig_to_b64(fig)

def rl_interval_chart(res):
    import matplotlib.pyplot as plt
    lo,hi=res["interval_low"],res["interval_high"]
    med=min(res["rl_median"],100)
    fig,ax=plt.subplots(figsize=(9.5,1.5))
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_yticks([])
    # zones
    ax.axvspan(0,5,color="#F87171",alpha=.18); ax.axvspan(5,20,color="#FBBF24",alpha=.16)
    ax.axvspan(20,50,color="#38BDF8",alpha=.14); ax.axvspan(50,100,color="#34D399",alpha=.14)
    ax.hlines(0,lo,hi,color="#22D3EE",lw=10,alpha=.6)
    ax.plot(med,0,"o",color="#EAF2FA",ms=20,zorder=5)
    ax.plot(med,0,"o",color="#22D3EE",ms=11,zorder=6)
    ax.set_xlim(0,100); ax.set_ylim(-1,1)
    ax.tick_params(colors="#9DB4C9",labelsize=14)
    ax.set_xticks([0,5,20,50,100])
    ax.set_xlabel("remaining life (years)",color="#9DB4C9",fontsize=14)
    return _fig_to_b64(fig)

def result_cards(res):
    is_insp=res["decision"]=="Inspect"
    verdict="Inspect" if is_insp else "No action"
    vcol=RED if is_insp else GREEN
    glow="251,113,133" if is_insp else "74,222,128"
    sub=("Predicted life is short; a check is warranted." if is_insp
         else "Condition reads as healthy.")
    icon="⚠️" if is_insp else "✓"
    LBL=(f'font-size:1.15rem;letter-spacing:1.4px;text-transform:uppercase;'
         f'color:{SOFT};font-weight:700;')
    CARD_H=200  # equal height for the two top cards
    c1,c2=st.columns(2, gap="medium")
    with c1:
        st.markdown(
            f'<div class="glass" style="border-color:rgba({glow},.45);height:{CARD_H}px;'
            f'display:flex;flex-direction:column;justify-content:center;text-align:left;'
            f'padding-left:1.8rem;">'
            f'<div style="{LBL}">Decision</div>'
            f'<div style="font-size:3.4rem;font-weight:900;color:{vcol};margin:.35rem 0;'
            f'text-shadow:0 0 26px rgba({glow},.6);">{icon} {verdict}</div>'
            f'<div style="color:{SOFT};font-size:1.22rem;">{sub}</div></div>',
            unsafe_allow_html=True)
    with c2:
        g=confidence_gauge(res["confidence"])
        st.markdown(
            f'<div class="glass" style="height:{CARD_H}px;display:flex;flex-direction:column;'
            f'justify-content:center;text-align:center;overflow:hidden;">'
            f'<div style="{LBL}">Confidence</div>'
            f'<img src="data:image/png;base64,{g}" style="height:112px;width:auto;'
            f'display:block;margin:.1rem auto 0;"/>'
            f'<div style="color:{SOFT};font-size:1.15rem;margin-top:.1rem;">'
            f'{res["fields_provided"]} of {res["fields_total"]} fields provided</div></div>',
            unsafe_allow_html=True)
    st.write("")
    ch=rl_interval_chart(res)
    st.markdown(
        f'<div class="glass">'
        f'<div style="{LBL}">Remaining life</div>'
        f'<div style="font-size:3.4rem;font-weight:900;color:{INK};margin:.2rem 0 .1rem;'
        f'text-shadow:0 0 22px rgba(34,211,238,.35);">{res["rl_display"]}'
        f'<span style="font-size:1.5rem;color:{SOFT};font-weight:600;"> years '
        f'(likely {res["interval_low"]:.0f} to {res["interval_high"]:.0f})</span></div>'
        f'<img src="data:image/png;base64,{ch}" style="width:100%;margin-top:.4rem;"/></div>',
        unsafe_allow_html=True)
    if res["coverage"]<0.6:
        st.caption("Filling more parameters will narrow the estimate.")

def bars_html(pairs,label_w=250):
    maxv=max(v for _,v in pairs); rows=""
    for name,v in pairs:
        pct=v/maxv*100
        rows+=(f'<div style="display:flex;align-items:center;margin:.6rem 0;">'
               f'<div style="width:{label_w}px;font-size:1.15rem;color:{SOFT};">{name}</div>'
               f'<div style="flex:1;background:#1a3350;border-radius:9px;height:28px;overflow:hidden;">'
               f'<div style="width:{pct:.0f}%;height:28px;border-radius:9px;'
               f'background:linear-gradient(90deg,#38BDF8,#2DD4BF);'
               f'box-shadow:0 0 14px rgba(56,189,248,.5);"></div></div>'
               f'<div style="width:70px;text-align:right;font-size:1.2rem;color:{INK};'
               f'font-weight:800;">{v*100:.1f}%</div></div>')
    return f'<div class="glass">{rows}</div>'


def sidebar_nav(active=""):
    """Custom sidebar navigation with large buttons."""
    logo=_logo_b64()
    with st.sidebar:
        if logo:
            st.markdown(
                f'<div class="sb-logo"><img src="data:image/png;base64,{logo}"/></div>',
                unsafe_allow_html=True)
        st.markdown('<div class="navtitle">Navigation</div>', unsafe_allow_html=True)
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_Single_pipeline.py", label="Single pipeline")
        st.page_link("pages/2_Fleet_analysis.py", label="Fleet analysis")
