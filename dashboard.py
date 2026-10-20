
from pathlib import Path
import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
DATA_FILE = ROOT / "data" / "raw" / "transactions.csv"

st.set_page_config(
    page_title="FINSIGHT | Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# FUTURISTIC DESIGN SYSTEM
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #050914;
    --panel: #0b1324;
    --cyan: #42e8ff;
    --green: #55ffbd;
    --muted: #91a4bd;
    --border: rgba(91, 214, 255, .19);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(0, 199, 255, .10), transparent 38%),
        radial-gradient(ellipse at 95% 25%, rgba(0, 255, 170, .055), transparent 30%),
        #050914;
    color: #e8f2ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(5, 9, 20, .75);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.6px;
    color: #f1f8ff;
}

h1 {
    font-size: clamp(2rem, 4vw, 3.1rem) !important;
    font-weight: 700 !important;
}

h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.1rem !important; }

p, label, [data-testid="stCaptionContainer"] {
    color: #a9bad0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1425 0%, #070c18 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f4fbff;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(15, 32, 53, .95), rgba(7, 15, 29, .96));
    border: 1px solid rgba(66, 232, 255, .22);
    border-radius: 16px;
    padding: 20px 20px 16px;
    box-shadow: 0 0 22px rgba(0, 200, 255, .045);
    transition: transform .25s ease, border-color .25s ease,
                box-shadow .25s ease;
    animation: riseIn .65s ease both;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(66, 232, 255, .65);
    box-shadow: 0 0 25px rgba(66, 232, 255, .13);
}

[data-testid="stMetricLabel"] {
    color: #91a4bd !important;
    font-size: .82rem !important;
}

[data-testid="stMetricValue"] {
    color: #f3fbff !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border: 1px solid rgba(66, 232, 255, .45);
    background: linear-gradient(100deg, rgba(66, 232, 255, .13),
                                        rgba(85, 255, 189, .09));
    color: #e9fbff;
    border-radius: 10px;
    transition: all .2s ease;
}

div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #55ffbd;
    color: #55ffbd;
    box-shadow: 0 0 18px rgba(85, 255, 189, .15);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #0b1526;
    border-color: #263e58;
    border-radius: 9px;
}

hr {
    border-color: rgba(66, 232, 255, .17);
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(66, 232, 255, .3);
    border-radius: 22px;
    padding: 30px 32px;
    margin: 5px 0 26px;
    background:
        radial-gradient(ellipse at 80% 0%, rgba(66, 232, 255, .15), transparent 45%),
        linear-gradient(120deg, rgba(13, 29, 50, .98), rgba(6, 15, 29, .96));
    box-shadow: 0 0 35px rgba(0, 200, 255, .055),
                inset 0 1px rgba(255,255,255,.035);
    animation: riseIn .7s ease both;
}

.hero:after {
    content: "";
    position: absolute;
    top: 0;
    left: -70%;
    width: 45%;
    height: 100%;
    background: linear-gradient(90deg, transparent,
                                rgba(66,232,255,.055), transparent);
    transform: skewX(-18deg);
    animation: sweep 8s ease-in-out infinite;
    pointer-events: none;
}

.eyebrow {
    color: #55ffbd;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: .72rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero-title {
    color: #f3fbff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4.4vw, 3.3rem);
    font-weight: 700;
    line-height: 1.12;
    margin: 0 0 12px;
}

.gradient-text {
    background: linear-gradient(90deg, #42e8ff, #55ffbd);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-description {
    max-width: 760px;
    line-height: 1.8;
    color: #9fb4cd;
    font-size: .96rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    border: 1px solid rgba(85, 255, 189, .3);
    background: rgba(85, 255, 189, .07);
    border-radius: 100px;
    padding: 8px 13px;
    color: #8dffd0;
    font-size: .76rem;
    margin-top: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #55ffbd;
    box-shadow: 0 0 12px #55ffbd;
    animation: pulse 1.7s infinite;
}

.section-label {
    color: #42e8ff;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    font-weight: 800;
    margin: 22px 0 8px;
}

.panel {
    background: linear-gradient(145deg, rgba(13, 27, 46, .96),
                                       rgba(8, 16, 30, .96));
    border: 1px solid rgba(91, 214, 255, .17);
    border-radius: 16px;
    padding: 19px 20px;
    margin: 8px 0 15px;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: #edf8ff;
    font-weight: 600;
    margin-bottom: 8px;
}

.panel-copy {
    color: #93a9c3;
    line-height: 1.65;
    font-size: .88rem;
}

.small-tag {
    color: #55ffbd;
    font-size: .7rem;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #637994;
    font-size: .76rem;
    padding: 25px 0 8px;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 12px #55ffbd; }
    50% { opacity: .45; box-shadow: 0 0 3px #55ffbd; }
}

@keyframes sweep {
    0%, 70% { left: -70%; }
    100% { left: 140%; }
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(9px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    *, *:before, *:after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA: historical evaluation snapshot
# =========================================================
DATASET_ROWS = 6_362_620
DATASET_FRAUD = 8_213

GLOBAL_POLICY = {
    "alerts": 20_852, "tp": 1_252, "fp": 19_600,
    "fn": 0, "tn": 17_127, "precision": 6.01, "recall": 100.00,
    "fp_cost": 98_000.00, "missed_cost": 0.00,
    "total_cost": 98_000.00,
}

SEGMENT_POLICY = {
    "alerts": 7_104, "tp": 1_226, "fp": 5_878,
    "fn": 26, "tn": 30_879, "precision": 17.26, "recall": 97.92,
    "fp_cost": 29_390.00, "missed_cost": 2_681_809.26,
    "total_cost": 2_711_199.26,
}


@st.cache_data
def load_json_reports():
    result = {}
    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*.json")):
            try:
                result[path.name] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                result[path.name] = {"report_read_error": str(exc)}
    return result


@st.cache_data
def load_markdown_reports():
    result = {}
    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*.md")):
            try:
                result[path.name] = path.read_text(encoding="utf-8")
            except OSError as exc:
                result[path.name] = f"Could not read report: {exc}"
    return result


@st.cache_data
def load_transaction_sample():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE, nrows=100_000)


json_reports = load_json_reports()
markdown_reports = load_markdown_reports()


# =========================================================
# REUSABLE COMPONENTS
# =========================================================
def hero(title, highlight, description, section="RISK INTELLIGENCE"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{section}</div>
        <div class="hero-title">{title}<br>
            <span class="gradient-text">{highlight}</span>
        </div>
        <div class="hero-description">{description}</div>
        <div class="status-pill">
            <span class="status-dot"></span>
            HISTORICAL EVALUATION · V1
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_label(text):
    st.markdown(
        f'<div class="section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def panel(title, description, tag=None):
    tag_html = (
        f'<div class="small-tag">{tag}</div>' if tag else ""
    )
    st.markdown(f"""
    <div class="panel">
        {tag_html}
        <div class="panel-title">{title}</div>
        <div class="panel-copy">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a9bad0", size=12),
        title=dict(font=dict(family="Space Grotesk", color="#eaf7ff", size=17)),
        margin=dict(l=12, r=12, t=65, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor="rgba(120,170,200,.10)",
            zerolinecolor="rgba(120,170,200,.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(120,170,200,.10)",
            zerolinecolor="rgba(120,170,200,.15)",
        ),
        hoverlabel=dict(
            bgcolor="#0b1728",
            bordercolor="#42e8ff",
            font_color="#eaf7ff",
        ),
    )
    return fig


def show_chart(fig, height=370):
    st.plotly_chart(
        style_chart(fig, height),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 22px">
        <div style="font-size:2.2rem">◈</div>
        <div style="font-family:'Space Grotesk';font-size:1.55rem;
                    font-weight:700;color:#f0fbff">
            FIN<span style="color:#42e8ff">SIGHT</span>
        </div>
        <div style="color:#829bb7;font-size:.76rem;letter-spacing:1.4px">
            FRAUD RISK INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="small-tag">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Risk Policy Comparison",
            "Business Cost",
            "Reports Explorer",
            "Transaction Explorer",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")
    st.markdown(
        '<span style="color:#55ffbd">●</span> Dashboard available',
        unsafe_allow_html=True,
    )
    st.caption("Local research environment")
    st.caption(f"{len(json_reports) + len(markdown_reports)} reports detected")

    if st.button("↻ Refresh report cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("FINSIGHT / V1.0")
    st.caption("Research demonstration — not live monitoring")


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    hero(
        "See the signal.",
        "Stop financial crime.",
        "A centralized view of transaction risk, fraud detection performance, "
        "and the business trade-offs behind every alert.",
        "FINTECH COMMAND CENTER",
    )

    st.info(
        "Snapshot notice: headline evaluation metrics are saved project "
        "results, not live model predictions."
    )

    fraud_rate = DATASET_FRAUD / DATASET_ROWS * 100

    section_label("01 / DATASET INTELLIGENCE")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", f"{DATASET_ROWS:,}")
    c2.metric("Fraud cases", f"{DATASET_FRAUD:,}")
    c3.metric("Fraud prevalence", f"{fraud_rate:.3f}%")
    c4.metric(
        "Research reports",
        f"{len(json_reports) + len(markdown_reports)}",
    )

    section_label("02 / DETECTION PERFORMANCE")
    left, right = st.columns(2)

    with left:
        panel(
            "Global threshold",
            "Catches all fraud cases in this historical test snapshot, "
            "but creates a very large alert workload.",
            "HIGH ALERT VOLUME",
        )
        a, b, c = st.columns(3)
        a.metric("Recall", f'{GLOBAL_POLICY["recall"]:.2f}%')
        b.metric("Precision", f'{GLOBAL_POLICY["precision"]:.2f}%')
        c.metric("Alerts", f'{GLOBAL_POLICY["alerts"]:,}')

    with right:
        panel(
            "Segment-specific policy",
            "Reduces alerts and improves precision, while missing some "
            "fraud cases in the evaluated period.",
            "LOWER ALERT VOLUME",
        )
        a, b, c = st.columns(3)
        a.metric("Recall", f'{SEGMENT_POLICY["recall"]:.2f}%')
        b.metric("Precision", f'{SEGMENT_POLICY["precision"]:.2f}%')
        c.metric("Alerts", f'{SEGMENT_POLICY["alerts"]:,}')

    section_label("03 / PERFORMANCE LANDSCAPE")
    chart_df = pd.DataFrame([
        {"Policy": "Global", "Metric": "Precision",
         "Value": GLOBAL_POLICY["precision"]},
        {"Policy": "Global", "Metric": "Recall",
         "Value": GLOBAL_POLICY["recall"]},
        {"Policy": "Segment-specific", "Metric": "Precision",
         "Value": SEGMENT_POLICY["precision"]},
        {"Policy": "Segment-specific", "Metric": "Recall",
         "Value": SEGMENT_POLICY["recall"]},
    ])

    fig = px.bar(
        chart_df,
        x="Policy",
        y="Value",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        title="Precision vs. recall · test snapshot",
        color_discrete_map={
            "Precision": "#42e8ff",
            "Recall": "#55ffbd",
        },
    )
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_yaxes(title="Percent", range=[0, 115])
    fig.update_xaxes(title="")
    show_chart(fig)

    panel(
        "Interpretation",
        "Recall measures how much fraud was detected. Precision measures "
        "how many flagged transactions were actually fraud. A high recall "
        "number alone does not tell us whether the alert workload is practical.",
        "ANALYST NOTE",
    )

    st.warning(
        "Historical test results are not a guarantee of future performance. "
        "The underlying dataset is simulated, and the displayed results "
        "should not be treated as production-bank evidence."
    )


# =========================================================
# PAGE 2: RISK POLICY COMPARISON
# =========================================================
elif page == "Risk Policy Comparison":
    hero(
        "Every alert has",
        "a business trade-off.",
        "Compare detection, false alarms, and missed fraud before selecting "
        "a policy for further validation.",
        "POLICY LAB",
    )

    comparison = pd.DataFrame([
        {
            "Policy": "Global threshold",
            "Alerts": GLOBAL_POLICY["alerts"],
            "True positives": GLOBAL_POLICY["tp"],
            "False positives": GLOBAL_POLICY["fp"],
            "Missed fraud": GLOBAL_POLICY["fn"],
            "Precision (%)": GLOBAL_POLICY["precision"],
            "Recall (%)": GLOBAL_POLICY["recall"],
        },
        {
            "Policy": "Segment-specific",
            "Alerts": SEGMENT_POLICY["alerts"],
            "True positives": SEGMENT_POLICY["tp"],
            "False positives": SEGMENT_POLICY["fp"],
            "Missed fraud": SEGMENT_POLICY["fn"],
            "Precision (%)": SEGMENT_POLICY["precision"],
            "Recall (%)": SEGMENT_POLICY["recall"],
        },
    ])

    section_label("POLICY SCORECARD")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section_label("ALERT OUTCOME BREAKDOWN")
    chart = comparison.melt(
        id_vars="Policy",
        value_vars=["True positives", "False positives", "Missed fraud"],
        var_name="Outcome",
        value_name="Transactions",
    )

    fig = px.bar(
        chart,
        x="Policy",
        y="Transactions",
        color="Outcome",
        barmode="group",
        title="Historical test-period outcomes",
        color_discrete_map={
            "True positives": "#55ffbd",
            "False positives": "#42e8ff",
            "Missed fraud": "#ff668e",
        },
    )
    show_chart(fig)

    panel(
        "Decision principle",
        "Do not select a threshold based on recall alone. Validate alert "
        "capacity, false-positive workload, missed-fraud losses, and "
        "performance on future time periods.",
        "RISK GOVERNANCE",
    )


# =========================================================
# PAGE 3: BUSINESS COST
# =========================================================
elif page == "Business Cost":
    hero(
        "Turn model metrics",
        "into business impact.",
        "Explore the estimated financial trade-offs under the project's "
        "illustrative false-positive and missed-fraud cost assumptions.",
        "COST INTELLIGENCE",
    )

    cost_df = pd.DataFrame([
        {
            "Policy": "Global threshold",
            "False-positive cost ($)": GLOBAL_POLICY["fp_cost"],
            "Missed-fraud cost ($)": GLOBAL_POLICY["missed_cost"],
            "Total estimated cost ($)": GLOBAL_POLICY["total_cost"],
        },
        {
            "Policy": "Segment-specific",
            "False-positive cost ($)": SEGMENT_POLICY["fp_cost"],
            "Missed-fraud cost ($)": SEGMENT_POLICY["missed_cost"],
            "Total estimated cost ($)": SEGMENT_POLICY["total_cost"],
        },
    ])

    section_label("ESTIMATED COST SCORECARD")
    st.dataframe(
        cost_df.style.format({
            "False-positive cost ($)": "${:,.2f}",
            "Missed-fraud cost ($)": "${:,.2f}",
            "Total estimated cost ($)": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="False-positive cost",
        x=cost_df["Policy"],
        y=cost_df["False-positive cost ($)"],
        marker_color="#42e8ff",
    ))
    fig.add_trace(go.Bar(
        name="Missed-fraud cost",
        x=cost_df["Policy"],
        y=cost_df["Missed-fraud cost ($)"],
        marker_color="#ff668e",
    ))
    fig.update_layout(
        barmode="stack",
        title="Illustrative estimated cost by policy",
        yaxis_title="Estimated cost (USD)",
    )
    show_chart(fig)

    st.warning(
        "These assumptions are illustrative, not verified bank economics. "
        "Actual fraud losses, investigation costs, recovery rates, and "
        "customer impact would need to be measured."
    )


# =========================================================
# PAGE 4: REPORTS EXPLORER
# =========================================================
elif page == "Reports Explorer":
    hero(
        "Evidence over",
        "assumptions.",
        "Inspect and download the JSON and Markdown reports generated "
        "throughout the fraud-risk research pipeline.",
        "RESEARCH ARCHIVE",
    )

    report_type = st.radio(
        "REPORT FORMAT",
        ["JSON", "Markdown"],
        horizontal=True,
    )

    if report_type == "JSON":
        if not json_reports:
            st.warning("No JSON reports found in the reports folder.")
        else:
            selected = st.selectbox(
                "Select report",
                list(json_reports.keys()),
            )
            st.json(json_reports[selected], expanded=2)
            st.download_button(
                "↓ Download JSON report",
                data=json.dumps(json_reports[selected], indent=2),
                file_name=selected,
                mime="application/json",
            )
    else:
        if not markdown_reports:
            st.warning("No Markdown reports found in the reports folder.")
        else:
            selected = st.selectbox(
                "Select report",
                list(markdown_reports.keys()),
            )
            st.markdown(markdown_reports[selected])
            st.download_button(
                "↓ Download Markdown report",
                data=markdown_reports[selected],
                file_name=selected,
                mime="text/markdown",
            )


# =========================================================
# PAGE 5: TRANSACTION EXPLORER
# =========================================================
elif page == "Transaction Explorer":
    hero(
        "Explore the",
        "transaction signal.",
        "Filter a sample of raw transactions and inspect amount patterns. "
        "The explorer reads a limited sample to keep your local machine responsive.",
        "TRANSACTION LAB",
    )

    df = load_transaction_sample()

    if df.empty:
        st.error(
            "Dataset not found at data/raw/transactions.csv. "
            "Please verify the file path."
        )
    else:
        st.success(f"Sample loaded: {len(df):,} rows (maximum 100,000).")

        if "type" in df.columns:
            types = sorted(df["type"].dropna().astype(str).unique())
            selected_types = st.multiselect(
                "Transaction types",
                types,
                default=types,
            )
            df = df[df["type"].astype(str).isin(selected_types)]

        if "isFraud" in df.columns:
            fraud_filter = st.selectbox(
                "Fraud status",
                ["All transactions", "Fraud only", "Non-fraud only"],
            )
            if fraud_filter == "Fraud only":
                df = df[df["isFraud"] == 1]
            elif fraud_filter == "Non-fraud only":
                df = df[df["isFraud"] == 0]

        a, b, c = st.columns(3)
        a.metric("Filtered transactions", f"{len(df):,}")

        if "isFraud" in df.columns:
            fraud_count = int(df["isFraud"].sum()) if len(df) else 0
            b.metric("Fraud cases in sample", f"{fraud_count:,}")
            c.metric(
                "Sample fraud rate",
                f"{fraud_count / len(df) * 100:.3f}%" if len(df) else "—",
            )

        if "amount" in df.columns and not df.empty:
            fig = px.histogram(
                df,
                x="amount",
                color="type" if "type" in df.columns else None,
                nbins=55,
                title="Transaction amount distribution",
                color_discrete_sequence=["#42e8ff", "#55ffbd", "#a78bfa"],
                opacity=.82,
            )
            show_chart(fig)

        st.dataframe(df.head(500), use_container_width=True)

        st.download_button(
            "↓ Download filtered sample",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_transaction_sample.csv",
            mime="text/csv",
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <span style="color:#42e8ff">◈ FINSIGHT</span>
    &nbsp; · &nbsp; FRAUD RISK INTELLIGENCE V1.0
    <br><br>
    Historical simulated data · Research demonstration · Not live monitoring
</div>
""", unsafe_allow_html=True)
