"""
Assessing Financial Assets' Market Value Using Data — Results Dashboard
MSc AI for Business Intelligence · University of Leicester
Mohamed Rizvi Shaik Abdulla (259042776)

Sprint 3 deliverable: interactive platform illustrating every model and experiment.
Design language: Geckoboard-style KPI tiles + Qlik-style sidebar & card layout.
Run with:  streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DATA = Path("dissertation_data")
PROC, FIG, NEWS = DATA / "processed", DATA / "figures", DATA / "news"

# palette
RED, BLUE, GREEN, AMBER = "#B5121B", "#3E6C9E", "#2E7D32", "#B8860B"
INK, MUTE, LINE, PANEL = "#1F2430", "#6B7280", "#E5E7EB", "#F7F8FA"
TICKERS = ["AAPL", "MSFT", "JPM", "TSLA", "XOM"]
SECTOR = {"AAPL":"Technology","MSFT":"Technology","JPM":"Financials","TSLA":"Consumer Disc.","XOM":"Energy"}

st.set_page_config(page_title="Asset-Value Modelling", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ============ GLOBAL STYLING ============
st.markdown(f"""
<style>
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1250px;}}
html, body, [class*="css"] {{color: {INK};}}
h1,h2,h3,h4 {{color:{INK}; font-weight:700;}}
section[data-testid="stSidebar"] {{background:{INK};}}
section[data-testid="stSidebar"] * {{color:#E9ECF2;}}
section[data-testid="stSidebar"] .brand {{border-left:4px solid {RED}; padding-left:.7rem;}}
/* keep the select/input readable: dark text on its white field (WCAG contrast) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{background:#fff !important;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input {{color:{INK} !important;
  -webkit-text-fill-color:{INK} !important;}}
ul[role="listbox"] li {{color:{INK} !important;}}
.stTabs [data-baseweb="tab-list"] {{gap:2px; border-bottom:1px solid {LINE};}}
.stTabs [data-baseweb="tab"] {{font-weight:600; padding:8px 14px;}}
.stTabs [aria-selected="true"] {{color:{RED}; border-bottom:3px solid {RED};}}

/* KPI tiles (Geckoboard style) */
.kpi {{background:#fff; border:1px solid {LINE}; border-radius:14px; padding:.85rem 1rem;
       box-shadow:0 1px 3px rgba(16,24,40,.04); height:150px; margin-bottom:1.1rem;}}
.kpi .lab {{font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:{MUTE}; font-weight:700;}}
.kpi .val {{font-size:2.0rem; font-weight:800; line-height:1.15; margin-top:.15rem;}}
.kpi .sub {{font-size:.8rem; color:{MUTE}; margin-top:.1rem;}}
.kpi.acc-red    {{border-top:4px solid {RED};}}
.kpi.acc-blue   {{border-top:4px solid {BLUE};}}
.kpi.acc-green  {{border-top:4px solid {GREEN};}}
.kpi.acc-amber  {{border-top:4px solid {AMBER};}}
.pill {{display:inline-block; font-size:.72rem; font-weight:700; padding:2px 9px;
        border-radius:999px; margin-left:.3rem;}}
.pill.g {{background:#E7F3E9; color:{GREEN};}}
.pill.r {{background:#FBEAEA; color:{RED};}}
.pill.n {{background:#EEF0F3; color:{MUTE};}}

/* content cards */
.card {{background:#fff; border:1px solid {LINE}; border-radius:14px; padding:1.15rem 1.35rem;
        box-shadow:0 1px 3px rgba(16,24,40,.04); margin-bottom:.8rem;}}
.card h4 {{margin:0 0 .5rem 0;}}
.big {{font-size:1.05rem;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(name):
    p = PROC / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_price(tk):
    return pd.read_csv(PROC / f"{tk}_daily_processed.csv", parse_dates=[0], index_col=0)

@st.cache_data
def load_sentiment(tk):
    n = pd.read_csv(NEWS / f"{tk}_news_scored.csv")
    ts = pd.to_datetime(n["created_at"], format="ISO8601", utc=True).dt.tz_convert("US/Eastern")
    n["date"] = ts.dt.tz_localize(None).dt.normalize()
    return n.groupby("date")["sent_score"].agg(sentiment="mean", n="size")

# ---- feature engineering (the same 15 features as the dissertation) ----
FEATURES = ["LogReturn_lag1","LogReturn_lag2","LogReturn_lag3","LogReturn_lag4","LogReturn_lag5",
            "SMA20_ratio","EMA20_ratio","RSI14","BollingerPctB","RollingVol10","VolumeChange",
            "Momentum10","IntradayRng","VolumeZ20","OvernightGap"]

def featurize(df):
    """OHLCV dataframe (Date index) -> 15 features + next-day-return target."""
    d = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["LogReturn"] = np.log(d["Close"]).diff()
    for l in range(1, 6):
        d[f"LogReturn_lag{l}"] = d["LogReturn"].shift(l)
    sma20 = d["Close"].rolling(20).mean(); ema20 = d["Close"].ewm(span=20, adjust=False).mean()
    d["SMA20_ratio"] = d["Close"]/sma20 - 1; d["EMA20_ratio"] = d["Close"]/ema20 - 1
    dl = d["Close"].diff()
    d["RSI14"] = 100 - 100/(1 + dl.clip(lower=0).rolling(14).mean()/(-dl.clip(upper=0)).rolling(14).mean())
    s20 = d["Close"].rolling(20).std()
    d["BollingerPctB"] = (d["Close"]-(sma20-2*s20))/(4*s20)
    d["RollingVol10"] = d["LogReturn"].rolling(10).std()
    d["VolumeChange"] = d["Volume"].pct_change()
    vm, vs = d["Volume"].rolling(20).mean(), d["Volume"].rolling(20).std()
    d["VolumeZ20"] = (d["Volume"]-vm)/vs
    d["Momentum10"] = d["Close"].pct_change(10)
    d["IntradayRng"] = (d["High"]-d["Low"])/d["Close"]
    d["OvernightGap"] = d["Open"]/d["Close"].shift(1) - 1
    d["Target"] = d["LogReturn"].shift(-1)
    return d

@st.cache_resource
def trained_model():
    """XGBoost trained on the pooled 5 development stocks (price-only, tuned config)."""
    import xgboost as xgb
    frames = []
    for tk in TICKERS:
        d = featurize(load_price(tk)).dropna(subset=FEATURES + ["Target"])
        frames.append(d)
    tr = pd.concat(frames)
    m = xgb.XGBRegressor(max_depth=2, n_estimators=200, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    m.fit(tr[FEATURES], tr["Target"])
    return m

def kpi(col, label, value, sub="", accent="blue", pill=None, pill_kind="n"):
    p = f'<span class="pill {pill_kind}">{pill}</span>' if pill else ""
    col.markdown(f"""<div class="kpi acc-{accent}"><div class="lab">{label}</div>
      <div class="val">{value}{p}</div><div class="sub">{sub}</div></div>""",
      unsafe_allow_html=True)

def spark(series, color=BLUE, h=60):
    f = go.Figure(go.Scatter(y=series, mode="lines", line=dict(color=color, width=1.8),
                             fill="tozeroy", fillcolor="rgba(62,108,158,.08)"))
    f.update_layout(height=h, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor="white",
                    paper_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f

def base_layout(fig, h=430, ytitle="", xtitle=""):
    fig.update_layout(height=h, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10,r=10,t=30,b=10), yaxis_title=ytitle, xaxis_title=xtitle,
                      font=dict(color=INK), legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor=LINE); fig.update_yaxes(gridcolor=LINE)
    return fig

# ============ SIDEBAR (Qlik-style filter rail) ============
with st.sidebar:
    st.markdown('<div class="brand"><h3 style="margin:.1rem 0;">Asset-Value<br>Modelling</h3>'
                '<div style="font-size:.8rem;opacity:.7;">Interactive Results Platform</div></div>',
                unsafe_allow_html=True)
    st.markdown("---")
    focus = st.selectbox("🔎 Focus stock", TICKERS,
                         help="Drives the Data Explorer and sparkline")
    st.caption(f"Sector: **{SECTOR[focus]}**")
    st.markdown("---")
    st.markdown("**Study at a glance**")
    st.markdown("- 5 development + 18 validation stocks\n- 2018–2024 daily data\n"
                "- 86,856 FinBERT-scored headlines\n- 9 model families · 7 signal channels")
    st.markdown("---")
    st.caption("Mohamed Rizvi Shaik Abdulla · 259042776\nUniversity of Leicester")

# ============ HEADER ============
st.markdown("### Assessing Financial Assets' Market Value Using Data")
st.caption("Can behavioural signals improve stock-price prediction?  ·  "
           "An efficient-markets investigation")

# ---- KPI strip ----
pf = load_price(focus)
last = pf["Close"].iloc[-1]; chg = pf["Close"].pct_change().iloc[-1]*100
k = st.columns(5)
kpi(k[0], "Headlines", "86.9k", "Benzinga · FinBERT", "blue")
kpi(k[1], "Models", "9", "naive → NN → ensembles", "blue")
kpi(k[2], "Best accuracy", "52.5%", "vs 52.5% naive", "red", pill="no edge", pill_kind="r")
kpi(k[3], "Volatility R²", "+0.12", "risk is forecastable", "green", pill="signal", pill_kind="g")
kpi(k[4], f"{focus} close", f"${last:,.0f}", f"{chg:+.1f}% on the day", "amber")

# ---- risk disclaimer: shown before anything else, as on any investing platform ----
st.markdown(f"""<div style="background:#FBF1F1;border-left:5px solid {RED};
     border-radius:0 8px 8px 0;padding:.75rem 1rem;margin:.2rem 0 1.1rem;">
  <div style="color:{RED};font-weight:700;font-size:.9rem;letter-spacing:.02em;">
    &#9888;&#65039; RESEARCH TOOL — NOT INVESTMENT ADVICE</div>
  <div style="color:{INK};font-size:.82rem;line-height:1.5;margin-top:.25rem;">
    These signals are frequently wrong. This study's central finding is that the models
    <b>do not beat a naive baseline</b>, and the apparent edge did not survive testing on unseen
    assets. Nothing here is a recommendation to buy or sell any security. Capital is at risk and
    past performance is not a guide to future performance.</div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["⚡ Live Signal", "Model Comparison", "Behaviour & Explainability",
                "Validation", "Data Explorer"])

# ============ TAB 1 — LIVE SIGNAL (interactive landing page) ============
with tabs[0]:
    left, right = st.columns([1, 2.1])
    with left:
        st.markdown("##### Run a signal")
        src = st.radio("Source", ["Ticker", "Upload CSV"], horizontal=True, label_visibility="collapsed")
        if src == "Ticker":
            tk_in = st.text_input("Symbol", value="NVDA", label_visibility="collapsed",
                                  placeholder="e.g. NVDA").strip().upper()
            if st.button("Generate signal ▶", use_container_width=True, type="primary") and tk_in:
                try:
                    import yfinance as yf
                    with st.spinner(f"Fetching {tk_in}…"):
                        raw = yf.download(tk_in, start="2018-01-01", progress=False, auto_adjust=True)
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    if len(raw) < 300: st.error("Not enough history.")
                    else: st.session_state["tryres"] = (raw, tk_in)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        else:
            up = st.file_uploader("OHLCV CSV", type=["csv"], label_visibility="collapsed")
            st.caption("Columns: Date, Open, High, Low, Close, Volume")
            if up is not None:
                try:
                    raw = pd.read_csv(up, parse_dates=[0], index_col=0)
                    raw.columns = [c.strip().title() for c in raw.columns]
                    if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns):
                        st.error("Missing OHLCV columns.")
                    else:
                        st.session_state["tryres"] = (raw.sort_index(), up.name.rsplit(".",1)[0].upper())
                except Exception as e:
                    st.error(f"Read failed: {e}")
        st.markdown(f'<div style="font-size:.75rem;color:{MUTE};margin-top:.6rem;">'
                    'Tuned XGBoost · 15 behavioural-technical features.</div>',
                    unsafe_allow_html=True)

        # ---- behavioural-state panel: what the model is reading right now ----
        if True:
            _df, _lab = st.session_state.get("tryres", (load_price(focus), focus))
            _d = featurize(_df).dropna(subset=FEATURES)
            if len(_d):
                cur = _d.iloc[-1]
                st.markdown("###### Behavioural state today")
                rsi = float(cur["RSI14"])
                zone = ("Overbought", RED) if rsi > 70 else (("Oversold", GREEN) if rsi < 30
                                                             else ("Neutral", MUTE))
                g = go.Figure(go.Indicator(
                    mode="gauge+number", value=rsi,
                    number={"font": {"size": 26, "color": INK}},
                    gauge={"axis": {"range": [0, 100], "tickvals": [0, 30, 50, 70, 100],
                                    "tickfont": {"size": 9}},
                           "bar": {"color": zone[1], "thickness": 0.7},
                           "bgcolor": "white", "borderwidth": 0,
                           "steps": [{"range": [0, 30], "color": "#E7F3E9"},
                                     {"range": [30, 70], "color": "#F3F4F6"},
                                     {"range": [70, 100], "color": "#FBEAEA"}]}))
                g.update_layout(height=150, margin=dict(l=18, r=18, t=6, b=0),
                                paper_bgcolor="white",
                                title={"text": f"RSI-14 · {zone[0]}", "font": {"size": 12},
                                       "x": 0.5, "y": 0.06})
                st.plotly_chart(g, use_container_width=True, config={"displayModeBar": False})

                # compact bars: how each behavioural driver sits vs its own history
                drivers = [("Momentum (10d)", "Momentum10", "herding"),
                           ("Trend vs 20d avg", "SMA20_ratio", "trend-following"),
                           ("Volatility (10d)", "RollingVol10", "panic / calm"),
                           ("Range position", "BollingerPctB", "over-extension")]
                rows, labs, cols_ = [], [], []
                for nm, col, _ in drivers:
                    s = _d[col].tail(252)
                    pct = float((s <= cur[col]).mean()) * 100      # percentile vs last year
                    rows.append(pct); labs.append(nm)
                    cols_.append(RED if pct > 80 else (GREEN if pct < 20 else BLUE))
                b = go.Figure(go.Bar(x=rows, y=labs, orientation="h", marker_color=cols_,
                        text=[f"{v:.0f}%" for v in rows], textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}<br>%{x:.0f}th percentile vs last year<extra></extra>"))
                b.add_vline(x=50, line_dash="dot", line_color=MUTE)
                b.update_layout(height=185, margin=dict(l=6, r=26, t=6, b=6), plot_bgcolor="white",
                                paper_bgcolor="white", xaxis=dict(range=[0, 108], visible=False),
                                font=dict(color=INK, size=11), showlegend=False)
                st.plotly_chart(b, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f'<div style="font-size:.72rem;color:{MUTE};margin-top:-.4rem;">'
                            'Percentile vs the last 12 months · dotted line = typical.</div>',
                            unsafe_allow_html=True)

    with right:
        if "tryres" not in st.session_state:
            st.session_state["tryres"] = (load_price(focus), focus)
        df, label = st.session_state["tryres"]
        d = featurize(df).dropna(subset=FEATURES).copy()
        d["pred"] = trained_model().predict(d[FEATURES])
        latest = d.iloc[-1]
        scored = d.dropna(subset=["Target"])
        acc   = float(np.mean(np.sign(scored["Target"]) == np.sign(scored["pred"])))*100
        naive = float((scored["Target"] > 0).mean())*100
        edge  = acc - naive
        up = latest["pred"] > 0
        st.markdown(f"##### {label} — next-day signal")
        m = st.columns(4)
        m[0].markdown(f'<div class="kpi acc-{"green" if up else "red"}"><div class="lab">Signal</div>'
            f'<div class="val" style="color:{GREEN if up else RED}">{"▲ UP" if up else "▼ DOWN"}</div>'
            f'<div class="sub">{latest["pred"]*100:+.2f}% predicted</div></div>', unsafe_allow_html=True)
        m[1].markdown(f'<div class="kpi acc-blue"><div class="lab">Accuracy</div>'
            f'<div class="val">{acc:.1f}%</div><div class="sub">{len(scored):,} days tested</div></div>',
            unsafe_allow_html=True)
        m[2].markdown(f'<div class="kpi acc-blue"><div class="lab">Naive baseline</div>'
            f'<div class="val">{naive:.1f}%</div><div class="sub">guess the trend</div></div>',
            unsafe_allow_html=True)
        m[3].markdown(f'<div class="kpi acc-{"green" if edge>0.5 else "red"}"><div class="lab">Edge</div>'
            f'<div class="val" style="color:{GREEN if edge>0.5 else RED}">{edge:+.1f}pp</div>'
            f'<div class="sub">{"beats naive" if edge>0.5 else "no real edge"}</div></div>',
            unsafe_allow_html=True)

        win = st.select_slider("History window", options=["6M","1Y","3Y","Max"], value="1Y")
        n = {"6M":126, "1Y":252, "3Y":756, "Max":len(d)}[win]
        dd = d.tail(n)
        f = go.Figure(go.Scatter(x=dd.index, y=dd["Close"], line=dict(color=INK, width=1.5),
                                 name="Close", hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>"))
        sig = dd[np.sign(dd["pred"]) != np.sign(dd["pred"].shift(1))]
        f.add_trace(go.Scatter(x=sig.index, y=sig["Close"], mode="markers", name="signal flip",
                    marker=dict(size=7, color=[GREEN if v>0 else RED for v in sig["pred"]],
                                line=dict(width=1, color="white")),
                    hovertemplate="%{x|%d %b %Y}<br>signal flip<extra></extra>"))
        base_layout(f, 330, ytitle="Price ($)")
        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
        st.caption("Markers show where the signal flips direction.")

# ============ TAB 2 — MODEL COMPARISON ============
with tabs[1]:
    st.markdown("#### Model tournament — every family, identical conditions")
    board = load_csv("model_tournament_summary.csv")
    if board is not None:
        board = board.rename(columns={board.columns[0]: "Model"}).sort_values("DirAcc")
        colors = [RED if m=="XGBoost (tuned)" else (BLUE if ("MLP" in m or "Forest" in m) else MUTE)
                  for m in board["Model"]]
        fig = go.Figure(go.Bar(x=board["DirAcc %"], y=board["Model"], orientation="h",
                     marker_color=colors, text=[f"{v:.1f}%" for v in board["DirAcc %"]],
                     textposition="outside", cliponaxis=False))
        fig.add_vline(x=52.5, line_dash="dash", line_color=INK,
                      annotation_text="naive baseline 52.5%", annotation_position="top right")
        base_layout(fig, 440, xtitle="Directional accuracy — avg of 15 tests")
        fig.update_xaxes(range=[48.5, 54])
        st.plotly_chart(fig, use_container_width=True)
        st.info("XGBoost & Random Forest lead; neural nets and linear models trail — the "
                "trees-and-nets-over-linear ranking of Gu, Kelly & Xiu (2020), reproduced here "
                "(XGB > Linear, Diebold–Mariano p = 0.001). But the dashed line is decisive: "
                "**no model beats guessing the market's upward trend.** The ceiling is set by the "
                "data (R² ≈ 0), not the algorithm.")
        with st.expander("Full metrics table (RMSE · MAE · directional accuracy)"):
            st.dataframe(board.set_index("Model").round(4), use_container_width=True)

# ============ TAB 3 — BEHAVIOURAL SIGNALS ============
with tabs[2]:
    st.markdown("#### Every behavioural signal tested vs the price-only baseline")
    ch = load_csv("behavioural_channels_summary.csv")
    if ch is not None:
        ch = ch.sort_values("delta_pp")
        colors = [RED if d<0 else GREEN for d in ch["delta_pp"]]
        fig = go.Figure(go.Bar(x=ch["delta_pp"], y=ch["channel"], orientation="h",
                     marker_color=colors, text=[f"{v:+.2f}pp" for v in ch["delta_pp"]],
                     textposition="outside", cliponaxis=False))
        fig.add_vline(x=0, line_color=INK)
        base_layout(fig, 400, xtitle="Change in accuracy vs baseline (pp) · 0 = no effect")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Every bar sits at zero or negative. Channels marked significant were significantly "
                "**worse** — features with no signal only add noise.")
    st.markdown("#### Why: news reflects the past, not the future")
    c1, c2 = st.columns(2)
    if (FIG/"sentiment_lead_lag.png").exists():
        c1.image(str(FIG/"sentiment_lead_lag.png"), caption="Sentiment tracks yesterday's return, not tomorrow's")
    if (FIG/"event_study_5min.png").exists():
        c2.image(str(FIG/"event_study_5min.png"), caption="5-min event study — price moves BEFORE the headline")

# ---- (continues in the Behaviour & Explainability tab) ----
with tabs[2]:
    st.markdown("#### What drives the model — SHAP")
    st.caption("The model leans on RSI, momentum, lagged returns and the overnight gap — the "
               "behavioural *footprints* in price. It also uses sentiment (10–26% of attribution) "
               "yet gains no accuracy: **importance ≠ predictive value.**")
    key = "price-only" if st.radio("Model", ["Price-only", "With sentiment"], horizontal=True)=="Price-only" else "with-sentiment"
    c1, c2 = st.columns(2)
    if (FIG/f"shap_beeswarm_AAPL_{key}.png").exists():
        c1.image(str(FIG/f"shap_beeswarm_AAPL_{key}.png"), caption="Per-day feature impact")
    if (FIG/f"shap_bar_AAPL_{key}.png").exists():
        c2.image(str(FIG/f"shap_bar_AAPL_{key}.png"), caption="Average importance")

# ============ TAB 4 — VALIDATION ============
with tabs[3]:
    st.markdown("#### The decisive test — 18 unseen stocks")
    val = load_csv("external_validation.csv")
    if val is not None:
        s = val.groupby("group").agg(DirAcc=("DirAcc","mean"), naive=("naive_up","mean")).reset_index()
        s["DirAcc"]*=100; s["naive"]*=100
        fig = go.Figure()
        fig.add_bar(x=s["group"], y=s["DirAcc"], name="My model", marker_color=RED,
                    text=[f"{v:.1f}%" for v in s["DirAcc"]], textposition="outside")
        fig.add_bar(x=s["group"], y=s["naive"], name="Naive (guess the trend)", marker_color=MUTE,
                    text=[f"{v:.1f}%" for v in s["naive"]], textposition="outside")
        base_layout(fig, 420, ytitle="Correct up/down calls").update_layout(barmode="group")
        fig.update_yaxes(range=[48, 54])
        st.plotly_chart(fig, use_container_width=True)
        st.warning("A small edge on the 5 development stocks **vanished** on 18 unseen stocks "
                   "(13,536 fresh days) — and on *both* sets the model merely matches the naive "
                   "baseline. The apparent edge was a small-sample artefact. Catching it is honest science.")
        with st.expander("Per-regime breakdown (directional accuracy %)"):
            st.dataframe((val.pivot_table(index="group", columns="regime", values="DirAcc")*100).round(2),
                         use_container_width=True)

# ============ TAB 5 — DATA EXPLORER ============
with tabs[4]:
    st.markdown(f"#### Explore — {focus}: price vs news sentiment")
    st.caption("Use the **Focus stock** selector in the sidebar to switch stocks.")
    yr = st.select_slider("Year", options=list(range(2018, 2025)), value=2022)
    p = pf[pf.index.year == yr]
    sent = load_sentiment(focus); s = sent[sent.index.year == yr]
    f1 = go.Figure(go.Scatter(x=p.index, y=p["Close"], line=dict(color=INK, width=1.6)))
    base_layout(f1, 300, ytitle="Price ($)").update_layout(showlegend=False)
    st.plotly_chart(f1, use_container_width=True)
    cols = [GREEN if v>0 else RED for v in s["sentiment"]]
    f2 = go.Figure(go.Bar(x=s.index, y=s["sentiment"], marker_color=cols))
    f2.add_hline(y=0, line_color=INK)
    base_layout(f2, 220, ytitle="Daily news sentiment")
    st.plotly_chart(f2, use_container_width=True)
    st.caption(f"{focus} · {yr} · {int(s['n'].sum())} headlines — sentiment reacts to price "
               "moves rather than leading them (the core finding, visible per-stock).")

# ---- findings summary, appended to the Validation tab ----
with tabs[3]:
    st.markdown("##### What the study found")
    c = st.columns(3)
    c[0].markdown(f"""<div class="card"><h4 style="color:{RED}">1 · No predictive edge</h4>
      7 behavioural channels · 9 models · 3 regimes — none significantly beat price alone.</div>""",
      unsafe_allow_html=True)
    c[1].markdown(f"""<div class="card"><h4 style="color:{RED}">2 · The market moves first</h4>
      Prices separate <i>before</i> headlines publish — news reports moves, it doesn't cause them.</div>""",
      unsafe_allow_html=True)
    c[2].markdown(f"""<div class="card"><h4 style="color:{GREEN}">3 · Risk is forecastable</h4>
      Direction R² ≈ 0, but volatility R² ≈ +0.12 — aim behavioural data at risk, not returns.</div>""",
      unsafe_allow_html=True)

    st.markdown("##### Headline numbers")
    t = pd.DataFrame({
        "Metric": ["Best model (XGBoost)", "Naive baseline", "Edge", "Direction R²",
                   "Volatility R²", "Sentiment effect (DM test)", "Held-out validation"],
        "Result": ["52.5%", "52.5%", "+0.0pp", "≈ 0", "+0.12", "p = 0.97 (null)", "edge did not replicate"],
        "Reading": ["directional accuracy, 15 tests", "always guess the trend", "not significant",
                    "explains ~none of direction", "explains 12% of move size",
                    "no improvement from news", "18 unseen stocks, 13,536 days"]})
    st.dataframe(t, use_container_width=True, hide_index=True)

    with st.expander("Method & references"):
        st.markdown("""
- **Design:** walk-forward validation, 3 expanding folds (2022 bear / 2023 recovery / 2024 bull);
  scalers fit on training windows only; no look-ahead.
- **Tests:** paired t-tests · Diebold–Mariano (1995) · Pesaran–Timmermann (1992).
- **Interpretation:** consistent with the semi-strong efficient-market hypothesis (Fama, 1970)
  and the low single-asset predictability of Gu, Kelly & Xiu (2020).
- **Sentiment:** FinBERT (Araci, 2019) on 86,856 Benzinga headlines, 16:00 ET cutoff.
        """)

st.markdown(f"""<hr style="border-color:{LINE}">
<div style="color:{MUTE};font-size:.78rem; line-height:1.5;">
<b>Data:</b> daily prices 2018–2024 (yfinance) · 86,856 headlines (Alpaca/Benzinga, FinBERT-scored) ·
insider &amp; institutional filings (SEC EDGAR). &nbsp;
<b>Key references:</b> Fama (1970); Gu, Kelly &amp; Xiu (2020); Araci (2019, FinBERT);
Diebold &amp; Mariano (1995); Pesaran &amp; Timmermann (1992). &nbsp;
<b>Note:</b> for research/educational use — not investment advice.<br>
Built with Streamlit + Plotly · © 2026 Mohamed Rizvi Shaik Abdulla · University of Leicester
</div>""", unsafe_allow_html=True)
