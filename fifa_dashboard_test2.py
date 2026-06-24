import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import warnings
import os
warnings.filterwarnings("ignore")

# Resolve paths relative to this script's location
_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Match Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — chalk-white pitch / gold / forest-green aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:     #FAF9F4;
    --bg2:    #F2F0E6;
    --card:   #FFFFFF;
    --gold:   #B8923A;
    --gold2:  #D4B56A;
    --slate:  #6B7280;
    --ink:    #1A1F1C;
    --muted:  #6B7280;
    --border: #E4E1D4;
    --green:  #134E4A;
    --green2: #0E3A37;
    --red:    #C2453B;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 2rem !important; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(184,146,58,0.10) 0%, transparent 70%);
}
.hero h1 {
    font-family: 'Montserrat', sans-serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.05;
    text-transform: uppercase;
}
.hero span { color: var(--gold); }
.hero p {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--slate);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2);
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--slate) !important;
    background: transparent;
    border-radius: 7px;
    padding: 0.5rem 1.1rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: var(--gold) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(26,31,28,0.04);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--green), transparent);
}
.kpi-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.kpi-value {
    font-family: 'Montserrat', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--ink);
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--gold);
    margin-top: 0.3rem;
    font-weight: 600;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--ink);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-left: 4px solid var(--gold);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* ── Goal type cards ── */
.goal-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.goal-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(26,31,28,0.04);
}
.goal-card .gc-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.goal-card .gc-label { font-family: 'Montserrat', sans-serif; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--slate); font-weight: 600; }
.goal-card .gc-val { font-family: 'Montserrat', sans-serif; font-size: 2rem; font-weight: 800; color: var(--gold); }
.goal-card .gc-pct { font-size: 0.78rem; color: var(--slate); margin-top: 0.2rem; }

/* ── Selectboxes ── */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
}
.stSelectbox label { font-family: 'Montserrat', sans-serif; font-size: 0.68rem !important; letter-spacing: 0.1em; text-transform: uppercase; color: var(--slate) !important; font-weight: 600 !important; }

/* Closed selectbox: selected value text + the typeahead search caret/text */
[data-baseweb="select"] {
    background: var(--card) !important;
}
[data-baseweb="select"] > div {
    background: var(--card) !important;
    color: var(--ink) !important;
}
[data-baseweb="select"] input {
    color: var(--ink) !important;
    caret-color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}
[data-baseweb="select"] svg { fill: var(--slate) !important; }

/* Dropdown list popover (rendered in a portal, so it needs its own rules) */
div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] div[data-baseweb="menu"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 14px rgba(26,31,28,0.12) !important;
}
div[data-baseweb="popover"] li[role="option"],
div[data-baseweb="popover"] [role="option"] * {
    background: var(--card) !important;
    color: var(--ink) !important;
}
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: var(--bg2) !important;
    color: var(--ink) !important;
}

/* ── Warning / info ── */
.pred-warn {
    font-size: 0.7rem;
    color: var(--muted);
    font-style: italic;
    margin-top: 0.5rem;
}

/* ── Prediction result card ── */
.pred-card {
    background: linear-gradient(135deg, var(--card), rgba(184,146,58,0.06));
    border: 1px solid var(--gold);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 1px 3px rgba(26,31,28,0.04);
}
.pred-winner {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--gold);
    text-transform: uppercase;
}
.pred-prob { font-size: 0.85rem; color: var(--slate); margin-top: 0.4rem; }

/* ── H2H bar ── */
.h2h-bar { display: flex; border-radius: 8px; overflow: hidden; height: 28px; margin: 0.5rem 0; }
.h2h-t1  { background: #B8923A; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; color:#fff; }
.h2h-draw { background: #E4E1D4; display:flex; align-items:center; justify-content:center; font-size:0.75rem; color:#6B7280; }
.h2h-t2  { background: #134E4A; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; color:#fff; }

/* Shootout top card */
.shoot-top {
    background: linear-gradient(135deg, rgba(184,146,58,0.10), rgba(19,78,74,0.08));
    border: 1px solid var(--gold);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(26,31,28,0.04);
}
.shoot-top .st-rank { font-family: 'Montserrat', sans-serif; font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold); font-weight: 700; }
.shoot-top .st-name { font-family: 'Montserrat', sans-serif; font-size: 2.2rem; font-weight: 900; color: var(--ink); text-transform: uppercase; }
.shoot-top .st-wins { font-size: 0.85rem; color: var(--slate); }

/* ── Model metrics cards (Prediction tab) — kept deliberately understated ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(26,31,28,0.03);
}
.metric-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.56rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--slate);
    font-weight: 600;
}
.metric-val {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--slate);
}

/* dataframe styling tweak so tables sit on white cards, not dark */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#FAF9F4"
BG2    = "#F2F0E6"
GOLD   = "#B8923A"
GOLD2  = "#D4B56A"
GOLD_LT = "#C7A555"  # subtle lighten of GOLD — used for bar-chart gradients so low bars stay visible
SLATE  = "#6B7280"
INK    = "#1A1F1C"
GREEN  = "#134E4A"
RED    = "#C2453B"
GRID   = "#E4E1D4"

def base_layout(**kw):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Montserrat, sans-serif", color=INK, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SLATE)),
        margin=dict(l=10, r=10, t=40, b=10),
        **kw
    )

def cap_first(s):
    """Capitalize only the first character of a label, leaving the rest (acronyms like 'USA') untouched."""
    return s[:1].upper() + s[1:] if isinstance(s, str) and s else s

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_results   = pd.read_csv(os.path.join(_DIR, "results.csv"))
    df_goals     = pd.read_csv(os.path.join(_DIR, "goalscorers.csv"))
    df_shootout  = pd.read_csv(os.path.join(_DIR, "shootouts.csv"))

    df_results["date"]   = pd.to_datetime(df_results["date"])
    df_results["winner"] = np.select(
        [df_results["home_score"] > df_results["away_score"],
         df_results["away_score"] > df_results["home_score"]],
        [df_results["home_team"], df_results["away_team"]],
        default="Draw"
    )
    df_results = df_results.dropna(subset=["home_score", "away_score"])
    return df_results, df_goals, df_shootout


@st.cache_data
def build_team_features(df_results):
    total_team_name = pd.unique(pd.concat([df_results["home_team"], df_results["away_team"]]))
    year_max  = df_results["date"].dt.year.max()
    df_4yr    = df_results[df_results["date"].dt.year >= year_max - 4]

    # goals ratio
    hg = df_4yr[["home_team","home_score","away_score"]].rename(columns={"home_team":"team","home_score":"scored","away_score":"conceded"})
    ag = df_4yr[["away_team","away_score","home_score"]].rename(columns={"away_team":"team","away_score":"scored","home_score":"conceded"})
    gs = pd.concat([hg,ag]).groupby("team").agg(total_scored=("scored","sum"), total_conceded=("conceded","sum")).reset_index()
    gs["goals_ratio"] = np.where(gs["total_conceded"]==0, gs["total_scored"], (gs["total_scored"]/gs["total_conceded"]).round(2))

    # overall win rate
    hw = df_4yr[["home_team","winner"]].rename(columns={"home_team":"team"}); hw["played"]=1; hw["won"]=(hw["winner"]==hw["team"]).astype(int)
    aw = df_4yr[["away_team","winner"]].rename(columns={"away_team":"team"}); aw["played"]=1; aw["won"]=(aw["winner"]==aw["team"]).astype(int)
    ov = pd.concat([hw,aw]).groupby("team").agg(total_played=("played","sum"), total_won=("won","sum")).reset_index()
    ov["overall_win_rate"] = (ov["total_won"]/ov["total_played"]).round(2)

    # home win rate
    ho = df_4yr[df_4yr["neutral"]==False][["home_team","winner"]].rename(columns={"home_team":"team"})
    ho["won"]=(ho["winner"]==ho["team"]).astype(int)
    hr = ho.groupby("team").agg(home_played=("won","count"), home_won=("won","sum")).reset_index()
    hr["home_win_rate"]=(hr["home_won"]/hr["home_played"]).round(2)

    # away win rate
    ao = df_4yr[df_4yr["neutral"]==False][["away_team","winner"]].rename(columns={"away_team":"team"})
    ao["won"]=(ao["winner"]==ao["team"]).astype(int)
    ar = ao.groupby("team").agg(away_played=("won","count"), away_won=("won","sum")).reset_index()
    ar["away_win_rate"]=(ar["away_won"]/ar["away_played"]).round(2)

    # neutral win rate
    no = df_4yr[df_4yr["neutral"]==True]
    hn = no[["home_team","winner"]].rename(columns={"home_team":"team"})
    an = no[["away_team","winner"]].rename(columns={"away_team":"team"})
    nc = pd.concat([hn,an]); nc["won"]=(nc["winner"]==nc["team"]).astype(int)
    nr = nc.groupby("team").agg(neutral_played=("won","count"), neutral_won=("won","sum")).reset_index()
    nr["neutral_win_rate"]=(nr["neutral_won"]/nr["neutral_played"]).round(2)

    # last 10
    ha = df_results[["date","home_team","winner"]].rename(columns={"home_team":"team"})
    aa = df_results[["date","away_team","winner"]].rename(columns={"away_team":"team"})
    am = pd.concat([ha,aa]).sort_values(["team","date"])
    def l10(g):
        last = g.tail(10); return round((last["winner"]==g.name).sum()/len(last),2)
    l10r = am.groupby("team").apply(l10).reset_index(); l10r.columns=["team","last10_win_rate"]

    tf = pd.DataFrame({"team": total_team_name})
    for part, cols in [
        (gs[["team","goals_ratio","total_scored","total_conceded"]], ["team","goals_ratio","total_scored","total_conceded"]),
        (ov[["team","overall_win_rate","total_played"]], ["team","overall_win_rate","total_played"]),
        (hr[["team","home_win_rate"]], ["team","home_win_rate"]),
        (ar[["team","away_win_rate"]], ["team","away_win_rate"]),
        (nr[["team","neutral_win_rate"]], ["team","neutral_win_rate"]),
        (l10r[["team","last10_win_rate"]], ["team","last10_win_rate"]),
    ]:
        tf = tf.merge(part[cols], on="team", how="left")
    tf = tf.fillna(0)
    return tf, df_4yr


@st.cache_resource
def train_model(df_4yr, team_features):
    """
    Trains the match-outcome model and also computes evaluation metrics
    (accuracy, precision, recall, F1, confusion matrix) on a held-out test
    split so the Prediction tab can surface model quality to the user.
    """
    rows = []
    for _, m in df_4yr.iterrows():
        t1, t2 = m["home_team"], m["away_team"]
        if t1 not in team_features["team"].values or t2 not in team_features["team"].values:
            continue
        s1 = team_features[team_features["team"]==t1].iloc[0]
        s2 = team_features[team_features["team"]==t2].iloc[0]
        rows.append({
            "t1_win_rate": s1["overall_win_rate"], "t2_win_rate": s2["overall_win_rate"],
            "t1_goals_ratio": s1["goals_ratio"],   "t2_goals_ratio": s2["goals_ratio"],
            "is_neutral": int(m["neutral"]),
            "result": "team1" if m["winner"]==t1 else ("team2" if m["winner"]==t2 else "draw")
        })
    td = pd.DataFrame(rows)
    X = td[["t1_win_rate","t2_win_rate","t1_goals_ratio","t2_goals_ratio","is_neutral"]]
    y = td["result"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    # ── evaluation metrics on the held-out test split ──────────────────────
    y_pred = model.predict(X_te)
    labels = model.classes_

    metrics = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_te, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels),
        "labels": list(labels),
        "report": classification_report(y_te, y_pred, zero_division=0, output_dict=True),
    }
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df_results, df_goals, df_shootout = load_data()
team_features, df_4yr = build_team_features(df_results)
model, model_metrics = train_model(df_4yr, team_features)

all_tournaments = sorted(df_results["tournament"].dropna().unique().tolist())
all_teams       = sorted(pd.unique(pd.concat([df_results["home_team"], df_results["away_team"]])).tolist())
all_countries   = sorted(df_results["country"].dropna().unique().tolist()) if "country" in df_results.columns else sorted(df_results["neutral"].unique().tolist())
all_years       = sorted(pd.to_datetime(df_goals["date"]).dt.year.dropna().unique().tolist(), reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚽ <span>FIFA</span> Match Analytics</h1>
    <p>International Football · Historical Data · Predictive Insights</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_team, tab_stats, tab_predict, tab_goals, tab_shootout = st.tabs([
    "🌐 Overview", "🏳️ Team", "📊 Stats", "🔮 Prediction", "⚽ Goals", "🥅 Shootouts"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    t_opts = ["All Tournaments"] + all_tournaments
    sel_t = st.selectbox("Tournament", t_opts, index=0, key="ov_tour")

    dff = df_results if sel_t == "All Tournaments" else df_results[df_results["tournament"] == sel_t]

    total_matches  = len(dff)
    total_goals    = int((dff["home_score"] + dff["away_score"]).sum())
    total_teams    = len(pd.unique(pd.concat([dff["home_team"], dff["away_team"]])))
    most_city      = dff["city"].value_counts().index[0] if len(dff) > 0 else "—"
    city_count     = int(dff["city"].value_counts().iloc[0]) if len(dff) > 0 else 0
    n_tournaments  = len(dff["tournament"].unique())

    home_c = dff.groupby("home_team").size()
    away_c = dff.groupby("away_team").size()
    mp     = home_c.add(away_c, fill_value=0)
    wc     = dff[dff["winner"] != "Draw"].groupby("winner").size()
    wr     = (wc / mp).dropna()
    eligible = mp[mp >= max(10, mp.quantile(0.3))].index
    wr_e   = wr[wr.index.isin(eligible)]
    best_team  = wr_e.idxmax() if len(wr_e) > 0 else "—"
    best_pct   = round(wr_e.max() * 100, 1) if len(wr_e) > 0 else 0
    worst_team = wr_e.idxmin() if len(wr_e) > 0 else "—"
    worst_pct  = round(wr_e.min() * 100, 1) if len(wr_e) > 0 else 0

    # KPI cards — row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Matches</div><div class="kpi-value">{total_matches:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Goals</div><div class="kpi-value">{total_goals:,}</div><div class="kpi-sub">{round(total_goals/max(total_matches,1),2)} per match</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Teams</div><div class="kpi-value">{total_teams}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Competitions</div><div class="kpi-value">{n_tournaments}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI cards — row 2
    c5, c6, c7 = st.columns(3)
    c5.markdown(f'<div class="kpi-card"><div class="kpi-label">Most Played City</div><div class="kpi-value" style="font-size:1.4rem">{most_city}</div><div class="kpi-sub">{city_count} matches</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="kpi-card"><div class="kpi-label">Most Successful Team</div><div class="kpi-value" style="font-size:1.3rem">{best_team}</div><div class="kpi-sub">{best_pct}% win rate</div></div>', unsafe_allow_html=True)
    c7.markdown(f'<div class="kpi-card"><div class="kpi-label">Least Successful Team</div><div class="kpi-value" style="font-size:1.3rem">{worst_team}</div><div class="kpi-sub">{worst_pct}% win rate</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Matches Over Time</div>', unsafe_allow_html=True)
    yr_count = dff.groupby(dff["date"].dt.year).size().reset_index(name="matches")
    yr_count.columns = ["year", "matches"]
    fig_yr = px.area(yr_count, x="year", y="matches", color_discrete_sequence=[GOLD])
    fig_yr.update_traces(fillcolor="rgba(184,146,58,0.15)", line_color=GOLD)
    fig_yr.update_xaxes(showgrid=False, color=GOLD, tickfont=dict(color=GOLD), title="Year", automargin=True)
    fig_yr.update_yaxes(showgrid=True, gridcolor=GRID, color=GOLD, tickfont=dict(color=GOLD), title="Matches", automargin=True)
    fig_yr.update_layout(**base_layout(height=300))
    st.plotly_chart(fig_yr, use_container_width=True, theme=None)

    st.markdown('<div class="section-title">Top 15 Teams by Matches Played</div>', unsafe_allow_html=True)
    top_teams = mp.sort_values(ascending=False).head(15).reset_index()
    top_teams.columns = ["team", "matches"]
    top_teams["team"] = top_teams["team"].apply(cap_first)
    fig_top = px.bar(top_teams, x="matches", y="team", orientation="h", color="matches",
                     color_continuous_scale=[[0, GOLD_LT],[1, GOLD]])
    fig_top.update_traces(cliponaxis=False)
    fig_top.update_layout(**base_layout(height=420), coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", color=GOLD, automargin=True,
                                     tickfont=dict(color=GOLD), ticklabelstandoff=10,
                                     title=dict(text="Team", standoff=20)),
                          xaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, automargin=True,
                                     tickfont=dict(color=GOLD),
                                     range=[0, top_teams["matches"].max()*1.15],
                                     title=dict(text="Matches", standoff=8)))
    st.plotly_chart(fig_top, use_container_width=True, theme=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEAM
# ════════════════════════════════════════════════════════════════════════════
with tab_team:
    col_a, col_b = st.columns(2)
    with col_a:
        t2_tour = st.selectbox("Tournament", ["All Tournaments"] + all_tournaments, key="team_tour")
    with col_b:
        t2_team = st.selectbox("Team", ["All Teams"] + all_teams, key="team_sel")

    dft = df_results.copy()
    if t2_tour != "All Tournaments":
        dft = dft[dft["tournament"] == t2_tour]
    if t2_team != "All Teams":
        dft = dft[(dft["home_team"] == t2_team) | (dft["away_team"] == t2_team)]

    if len(dft) == 0:
        st.info("No matches found for this selection.")
    else:
        if t2_team != "All Teams":
            team = t2_team
            home_m = dft[dft["home_team"] == team]
            away_m = dft[dft["away_team"] == team]

            home_wins  = (home_m["winner"] == team).sum()
            home_draws = (home_m["winner"] == "Draw").sum()
            home_loss  = len(home_m) - home_wins - home_draws
            away_wins  = (away_m["winner"] == team).sum()
            away_draws = (away_m["winner"] == "Draw").sum()
            away_loss  = len(away_m) - away_wins - away_draws

            total_w = home_wins + away_wins
            total_d = home_draws + away_draws
            total_l = home_loss + away_loss

            cl, cr = st.columns(2)
            with cl:
                st.markdown('<div class="section-title">Home vs Away Wins</div>', unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=["Home Wins", "Away Wins"],
                    values=[home_wins, away_wins],
                    hole=0.55,
                    marker_colors=[GOLD, GREEN],
                    textfont_color=INK,
                ))
                fig_pie.update_layout(**base_layout(height=300), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True, theme=None)

            with cr:
                st.markdown('<div class="section-title">Win / Draw / Loss</div>', unsafe_allow_html=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name="Wins",   x=["Result"], y=[total_w], marker_color=GOLD))
                fig_bar.add_trace(go.Bar(name="Draws",  x=["Result"], y=[total_d], marker_color=SLATE))
                fig_bar.add_trace(go.Bar(name="Losses", x=["Result"], y=[total_l], marker_color=RED))
                fig_bar.update_layout(**base_layout(height=300), barmode="group",
                                      xaxis=dict(color=GOLD, tickfont=dict(color=GOLD), title=dict(text="Result", standoff=8), automargin=True),
                                      yaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, tickfont=dict(color=GOLD), title=dict(text="Count", standoff=8), automargin=True))
                st.plotly_chart(fig_bar, use_container_width=True, theme=None)

            # form — last 10
            st.markdown('<div class="section-title">Recent Form (Last 10 Matches)</div>', unsafe_allow_html=True)
            ha_all = pd.concat([
                dft[["date","home_team","away_team","winner"]].rename(columns={"home_team":"t"}),
                dft[["date","home_team","away_team","winner"]].rename(columns={"away_team":"t"})
            ])
            recent = ha_all[ha_all["t"] == team].sort_values("date").tail(10)
            badges = ""
            for _, row in recent.iterrows():
                if row["winner"] == team:
                    badges += '<span style="background:#B8923A;color:#fff;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;margin:2px;font-family:Montserrat,sans-serif;">W</span> '
                elif row["winner"] == "Draw":
                    badges += '<span style="background:#E4E1D4;color:#6B7280;padding:3px 10px;border-radius:4px;font-size:0.75rem;margin:2px;font-family:Montserrat,sans-serif;">D</span> '
                else:
                    badges += '<span style="background:#F7DEDC;color:#C2453B;padding:3px 10px;border-radius:4px;font-size:0.75rem;margin:2px;font-family:Montserrat,sans-serif;">L</span> '
            st.markdown(f'<div style="margin:0.5rem 0">{badges}</div>', unsafe_allow_html=True)

        else:
            # aggregate across all teams
            st.markdown('<div class="section-title">Top 10 Teams by Wins</div>', unsafe_allow_html=True)
            wins_all = dft[dft["winner"] != "Draw"]["winner"].value_counts().head(10).reset_index()
            wins_all.columns = ["team", "wins"]
            wins_all["team"] = wins_all["team"].apply(cap_first)
            fig_wa = px.bar(wins_all, x="wins", y="team", orientation="h",
                            color="wins", color_continuous_scale=[[0,GOLD_LT],[1,GOLD]])
            fig_wa.update_traces(cliponaxis=False)
            fig_wa.update_layout(**base_layout(height=400), coloraxis_showscale=False,
                                 yaxis=dict(autorange="reversed", color=GOLD, automargin=True,
                                            tickfont=dict(color=GOLD), ticklabelstandoff=10,
                                            title=dict(text="Team", standoff=20)),
                                 xaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, automargin=True,
                                            tickfont=dict(color=GOLD),
                                            range=[0, wins_all["wins"].max()*1.15],
                                            title=dict(text="Wins", standoff=8)))
            st.plotly_chart(fig_wa, use_container_width=True, theme=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATS (Head to Head)
# ════════════════════════════════════════════════════════════════════════════
with tab_stats:
    time_range_opts = {"Past 1 Year":1,"Past 2 Years":2,"Past 5 Years":5,
                       "Past 10 Years":10,"Past 20 Years":20,"Past 50 Years":50,
                       "Past 100 Years":100,"All Time":None}

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_t1 = st.selectbox("Team 1", all_teams, key="h2h_t1")
    with sc2:
        default_t2 = all_teams[1] if len(all_teams) > 1 else all_teams[0]
        s_t2 = st.selectbox("Team 2", all_teams, index=1, key="h2h_t2")
    with sc3:
        s_range = st.selectbox("Time Range", list(time_range_opts.keys()), index=4, key="h2h_range")

    years = time_range_opts[s_range]
    dfs = df_results.copy()
    if years:
        cutoff = dfs["date"].dt.year.max() - years
        dfs = dfs[dfs["date"].dt.year >= cutoff]

    h2h = dfs[
        ((dfs["home_team"]==s_t1) & (dfs["away_team"]==s_t2)) |
        ((dfs["home_team"]==s_t2) & (dfs["away_team"]==s_t1))
    ].copy()

    if len(h2h) == 0:
        st.info(f"No head-to-head matches found between {s_t1} and {s_t2} in this time range.")
    else:
        t1_wins  = (h2h["winner"] == s_t1).sum()
        t2_wins  = (h2h["winner"] == s_t2).sum()
        draws    = (h2h["winner"] == "Draw").sum()
        total_h2 = len(h2h)

        t1_goals = int(h2h.apply(lambda r: r["home_score"] if r["home_team"]==s_t1 else r["away_score"], axis=1).sum())
        t2_goals = int(h2h.apply(lambda r: r["home_score"] if r["home_team"]==s_t2 else r["away_score"], axis=1).sum())

        # best win
        def goal_diff(row, team):
            if row["home_team"] == team:
                return row["home_score"] - row["away_score"]
            else:
                return row["away_score"] - row["home_score"]

        t1_matches = h2h[h2h["winner"]==s_t1].copy()
        t2_matches = h2h[h2h["winner"]==s_t2].copy()

        if len(t1_matches):
            t1_matches["gd"] = t1_matches.apply(lambda r: goal_diff(r, s_t1), axis=1)
            best_t1 = t1_matches.loc[t1_matches["gd"].idxmax()]
            best_t1_str = f"{int(best_t1['home_score'])}–{int(best_t1['away_score'])} ({best_t1['home_team']} vs {best_t1['away_team']})"
        else:
            best_t1_str = "No wins"

        if len(t2_matches):
            t2_matches["gd"] = t2_matches.apply(lambda r: goal_diff(r, s_t2), axis=1)
            best_t2 = t2_matches.loc[t2_matches["gd"].idxmax()]
            best_t2_str = f"{int(best_t2['home_score'])}–{int(best_t2['away_score'])} ({best_t2['home_team']} vs {best_t2['away_team']})"
        else:
            best_t2_str = "No wins"

        # H2H bar
        t1w = round(t1_wins/total_h2*100)
        dw  = round(draws/total_h2*100)
        t2w = 100 - t1w - dw
        st.markdown(f"""
        <div class="section-title">Head-to-Head · {total_h2} matches</div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{SLATE};margin-bottom:4px;">
            <span>{s_t1}</span><span>Draw</span><span>{s_t2}</span>
        </div>
        <div class="h2h-bar">
            <div class="h2h-t1"  style="width:{t1w}%">{t1_wins}</div>
            <div class="h2h-draw" style="width:{dw}%">{draws}</div>
            <div class="h2h-t2"  style="width:{t2w}%">{t2_wins}</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:{SLATE};margin-top:4px;">
            <span>{t1w}%</span><span>{dw}%</span><span>{t2w}%</span>
        </div>
        """, unsafe_allow_html=True)

        sc4, sc5, sc6 = st.columns(3)
        sc4.markdown(f'<div class="kpi-card"><div class="kpi-label">Goals — {s_t1}</div><div class="kpi-value">{t1_goals}</div></div>', unsafe_allow_html=True)
        sc5.markdown(f'<div class="kpi-card"><div class="kpi-label">Goals — {s_t2}</div><div class="kpi-value">{t2_goals}</div></div>', unsafe_allow_html=True)
        sc6.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Matches</div><div class="kpi-value">{total_h2}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sc7, sc8 = st.columns(2)
        sc7.markdown(f'<div class="kpi-card"><div class="kpi-label">Best Win — {s_t1}</div><div class="kpi-value" style="font-size:1rem;margin-top:4px">{best_t1_str}</div></div>', unsafe_allow_html=True)
        sc8.markdown(f'<div class="kpi-card"><div class="kpi-label">Best Win — {s_t2}</div><div class="kpi-value" style="font-size:1rem;margin-top:4px">{best_t2_str}</div></div>', unsafe_allow_html=True)

        # Win % over time line chart
        st.markdown('<div class="section-title">Win Rate Over Time</div>', unsafe_allow_html=True)
        h2h["year"] = h2h["date"].dt.year
        yr_grp = h2h.groupby("year").apply(lambda g: pd.Series({
            s_t1: round((g["winner"]==s_t1).mean()*100, 1),
            "Draw": round((g["winner"]=="Draw").mean()*100, 1),
            s_t2: round((g["winner"]==s_t2).mean()*100, 1),
        })).reset_index()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=yr_grp["year"], y=yr_grp[s_t1], name=s_t1, line=dict(color=GOLD, width=2), mode="lines+markers"))
        fig_line.add_trace(go.Scatter(x=yr_grp["year"], y=yr_grp["Draw"], name="Draw", line=dict(color=SLATE, width=1.5, dash="dot"), mode="lines+markers"))
        fig_line.add_trace(go.Scatter(x=yr_grp["year"], y=yr_grp[s_t2], name=s_t2, line=dict(color=GREEN, width=2), mode="lines+markers"))
        fig_line.update_xaxes(showgrid=False, color=GOLD, tickfont=dict(color=GOLD), title="Year", automargin=True)
        fig_line.update_yaxes(showgrid=True, gridcolor=GRID, color=GOLD, tickfont=dict(color=GOLD), title="Win %", automargin=True)
        fig_line.update_layout(**base_layout(height=320))
        st.plotly_chart(fig_line, use_container_width=True, theme=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-title">Match Outcome Predictor</div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_t1 = st.selectbox("Team 1 (Home)", all_teams, key="pred_t1")
    with pc2:
        p_t2 = st.selectbox("Team 2 (Away)", all_teams, index=1, key="pred_t2")
    with pc3:
        venue_opts = ["Neutral Venue", "Team 1's Country", "Team 2's Country"]
        p_venue = st.selectbox("Venue", venue_opts, key="pred_venue")

    is_neutral = 1 if p_venue == "Neutral Venue" else 0

    if st.button("Predict Outcome ↗", key="pred_btn"):
        if p_t1 == p_t2:
            st.warning("Please select two different teams.")
        elif p_t1 not in team_features["team"].values or p_t2 not in team_features["team"].values:
            st.warning("Insufficient data for one or both teams.")
        else:
            s1 = team_features[team_features["team"]==p_t1].iloc[0]
            s2 = team_features[team_features["team"]==p_t2].iloc[0]
            X_pred = pd.DataFrame([{
                "t1_win_rate": s1["overall_win_rate"], "t2_win_rate": s2["overall_win_rate"],
                "t1_goals_ratio": s1["goals_ratio"],   "t2_goals_ratio": s2["goals_ratio"],
                "is_neutral": is_neutral
            }])
            probs  = model.predict_proba(X_pred)[0]
            labels = model.classes_
            prob_map = dict(zip(labels, probs))

            t1_p   = round(prob_map.get("team1", 0)*100, 1)
            draw_p = round(prob_map.get("draw", 0)*100, 1)
            t2_p   = round(prob_map.get("team2", 0)*100, 1)
            winner_label = p_t1 if t1_p >= t2_p and t1_p >= draw_p else (p_t2 if t2_p >= draw_p else "Draw")

            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:0.7rem;letter-spacing:0.14em;text-transform:uppercase;color:{SLATE};margin-bottom:0.5rem">Predicted Outcome</div>
                <div class="pred-winner">{winner_label}</div>
                <div class="pred-prob">{p_t1} {t1_p}% &nbsp;·&nbsp; Draw {draw_p}% &nbsp;·&nbsp; {p_t2} {t2_p}%</div>
            </div>
            """, unsafe_allow_html=True)

            fig_prob = go.Figure(go.Bar(
                x=[cap_first(p_t1), "Draw", cap_first(p_t2)], y=[t1_p, draw_p, t2_p],
                marker_color=[GOLD, SLATE, GREEN],
                text=[f"{v}%" for v in [t1_p, draw_p, t2_p]],
                textposition="outside", textfont_color=GOLD
            ))
            fig_prob.update_traces(cliponaxis=False)
            fig_prob.update_layout(**base_layout(height=280),
                                   yaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, tickfont=dict(color=GOLD), range=[0,108],
                                              title=dict(text="Win Probability (%)", standoff=8), automargin=True),
                                   xaxis=dict(color=GOLD, tickfont=dict(color=GOLD), title=dict(text="Outcome", standoff=8), automargin=True))
            st.plotly_chart(fig_prob, use_container_width=True, theme=None)

    st.markdown(f'<div class="pred-warn">⚠ Disclaimer: This prediction is generated by a logistic regression model trained on historical match data. Results should be treated as indicative only and not used for wagering or official purposes.</div>', unsafe_allow_html=True)

    # ── Model performance section ───────────────────────────────────────────
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f'<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-val">{model_metrics["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val">{model_metrics["precision"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-val">{model_metrics["recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    mc4.markdown(f'<div class="metric-card"><div class="metric-label">F1 Score</div><div class="metric-val">{model_metrics["f1"]*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="pred-warn">Metrics computed on a held-out 20% test split (weighted average across draw / team1 / team2 classes).</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — GOALS
# ════════════════════════════════════════════════════════════════════════════
with tab_goals:
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        g_tour = st.selectbox("Tournament", ["All Tournaments"] + all_tournaments, key="goals_tour")
    with gc2:
        g_team = st.selectbox("Team", ["All Teams"] + all_teams, key="goals_team")
    with gc3:
        g_year = st.selectbox("Year", ["All Years"] + all_years, key="goals_year")

    dfg = df_goals.copy()
    if g_tour != "All Tournaments":
        tour_dates = df_results[df_results["tournament"]==g_tour]["date"].dt.date
        dfg = dfg[pd.to_datetime(dfg["date"]).dt.date.isin(tour_dates)]
    if g_team != "All Teams":
        dfg = dfg[dfg["team"] == g_team] if "team" in dfg.columns else dfg[dfg["scorer"].notna()]
    if g_year != "All Years":
        dfg = dfg[pd.to_datetime(dfg["date"]).dt.year == g_year]

    goal_fhalf  = (dfg["minute"] <= 45).sum()
    goal_shalf  = ((dfg["minute"] > 45) & (dfg["minute"] <= 90)).sum()
    goal_extra  = (dfg["minute"] > 90).sum()
    total_g     = len(dfg)

    open_play = ((dfg["own_goal"]==False) & (dfg["penalty"]==False)).sum()
    own_goals  = (dfg["own_goal"]==True).sum()
    penalties  = (dfg["penalty"]==True).sum()

    # Donut
    col_donut, col_cards = st.columns([1, 1])
    with col_donut:
        st.markdown('<div class="section-title">Goals by Period</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["First Half", "Second Half", "Extra Time"],
            values=[goal_fhalf, goal_shalf, goal_extra],
            hole=0.6,
            marker_colors=[GOLD, GREEN, RED],
        ))
        fig_donut.update_layout(**base_layout(height=300))
        st.plotly_chart(fig_donut, use_container_width=True, theme=None)

    with col_cards:
        st.markdown('<div class="section-title">Goal Types</div>', unsafe_allow_html=True)
        op_pct = round(open_play/max(total_g,1)*100,1)
        og_pct = round(own_goals/max(total_g,1)*100,1)
        pe_pct = round(penalties/max(total_g,1)*100,1)
        st.markdown(f"""
        <div class="goal-cards">
            <div class="goal-card">
                <div class="gc-icon">🎯</div>
                <div class="gc-label">Open Play</div>
                <div class="gc-val">{open_play:,}</div>
                <div class="gc-pct">{op_pct}% of all goals</div>
            </div>
            <div class="goal-card">
                <div class="gc-icon">😬</div>
                <div class="gc-label">Own Goals</div>
                <div class="gc-val">{own_goals:,}</div>
                <div class="gc-pct">{og_pct}% of all goals</div>
            </div>
            <div class="goal-card">
                <div class="gc-icon">⚡</div>
                <div class="gc-label">Penalties</div>
                <div class="gc-val">{penalties:,}</div>
                <div class="gc-pct">{pe_pct}% of all goals</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Top scorers (no own goals)
    st.markdown('<div class="section-title">Top 5 Goal Scorers</div>', unsafe_allow_html=True)
    clean_goals = dfg[dfg["own_goal"]==False]
    top5 = clean_goals["scorer"].value_counts().head(5).reset_index()
    top5.columns = ["scorer", "goals"]
    top5["scorer"] = top5["scorer"].apply(cap_first)
    fig_top5 = px.bar(top5, x="goals", y="scorer", orientation="h",
                      color="goals", color_continuous_scale=[[0,GOLD_LT],[1,GOLD]],
                      text="goals")
    fig_top5.update_traces(textfont_color=GOLD, textposition="outside", cliponaxis=False)
    fig_top5.update_layout(**base_layout(height=300), coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed", color=GOLD, automargin=True,
                                      tickfont=dict(color=GOLD), ticklabelstandoff=10,
                                      title=dict(text="Scorer", standoff=20)),
                           xaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, automargin=True,
                                      tickfont=dict(color=GOLD),
                                      range=[0, top5["goals"].max()*1.2],
                                      title=dict(text="Goals", standoff=8)))
    st.plotly_chart(fig_top5, use_container_width=True, theme=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — SHOOTOUTS
# ════════════════════════════════════════════════════════════════════════════
with tab_shootout:
    shootouts_known = df_shootout.dropna(subset=["first_shooter"])
    shootouts_known = shootouts_known.copy()
    shootouts_known["first_shooter_won"] = shootouts_known["first_shooter"] == shootouts_known["winner"]
    first_win_pct = round(shootouts_known["first_shooter_won"].mean() * 100, 1)

    top5_shoot = df_shootout["winner"].value_counts().head(5).reset_index()
    top5_shoot.columns = ["team", "wins"]
    top5_shoot["team"] = top5_shoot["team"].apply(cap_first)

    rank1 = top5_shoot.iloc[0]

    # Top card
    st.markdown(f"""
    <div class="shoot-top">
        <div class="st-rank">🏆 Most Shootout Wins</div>
        <div class="st-name">{rank1['team']}</div>
        <div class="st-wins">{int(rank1['wins'])} shootout wins</div>
    </div>
    """, unsafe_allow_html=True)

    col_sh1, col_sh2 = st.columns(2)

    with col_sh1:
        st.markdown('<div class="section-title">First Shooter Win %</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=first_win_pct,
            number={"suffix":"%", "font":{"color":GOLD,"family":"Montserrat","size":36}},
            gauge={
                "axis": {"range":[0,100], "tickcolor":SLATE},
                "bar":  {"color": GOLD},
                "bgcolor": BG2,
                "bordercolor": GRID,
                "steps": [{"range":[0,50],"color":BG2},{"range":[50,100],"color":"#EDEAD9"}],
                "threshold": {"line":{"color":INK,"width":2},"thickness":0.75,"value":50}
            }
        ))
        fig_gauge.update_layout(**base_layout(height=280))
        st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

    with col_sh2:
        st.markdown('<div class="section-title">Top 5 — Most Shootout Wins</div>', unsafe_allow_html=True)
        top5_shoot["rank"] = range(1, len(top5_shoot)+1)
        fig_sh = px.bar(top5_shoot, x="wins", y="team", orientation="h",
                        color="wins", color_continuous_scale=[[0,GOLD_LT],[1,GOLD]],
                        text="wins")
        fig_sh.update_traces(textfont_color=GOLD, textposition="outside", cliponaxis=False)
        fig_sh.update_layout(**base_layout(height=280), coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed", color=GOLD, automargin=True,
                                        tickfont=dict(color=GOLD), ticklabelstandoff=10,
                                        title=dict(text="Team", standoff=20)),
                             xaxis=dict(showgrid=True, gridcolor=GRID, color=GOLD, automargin=True,
                                        tickfont=dict(color=GOLD),
                                        range=[0, top5_shoot["wins"].max()*1.2],
                                        title=dict(text="Wins", standoff=8)))
        st.plotly_chart(fig_sh, use_container_width=True, theme=None)
