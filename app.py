"""
Enhanced UPI Fraud Detection System — Streamlit App
Uses AI/ML ensemble: Naive Bayes + Decision Tree + Rule-Based Scoring
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from fraud_detection import detect_fraud

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 600; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 0.95rem; opacity: 0.7; margin: 0.4rem 0 0; }

    .model-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .model-label { font-size: 0.78rem; font-weight: 600; color: #5c6bc0; text-transform: uppercase; letter-spacing: 0.5px; }
    .model-score { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #1a237e; }
    .model-desc  { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }

    .risk-high   { background: #fff0f0; border: 1.5px solid #ff4444; border-radius: 12px; padding: 1.2rem; }
    .risk-medium { background: #fff8e1; border: 1.5px solid #ffa000; border-radius: 12px; padding: 1.2rem; }
    .risk-low    { background: #e8f5e9; border: 1.5px solid #43a047; border-radius: 12px; padding: 1.2rem; }

    .feature-pill {
        display: inline-block;
        background: #ede7f6;
        color: #4527a0;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .bar-container { background: #e8eaf6; border-radius: 6px; height: 10px; margin-top: 4px; }
    .bar-fill { height: 10px; border-radius: 6px; transition: width 0.5s; }

    .ensemble-badge {
        background: linear-gradient(90deg, #6a1b9a, #283593);
        color: white;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
    }
    .stTextArea textarea { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.95rem; }
    div[data-testid="metric-container"] { background: #f0f4ff; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ UPI Fraud Shield</h1>
  <p>AI/ML Ensemble Detection · Naive Bayes · Decision Tree · Feature Engineering · Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: About the Models ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI/ML Models Used")
    st.markdown("""
    This system uses an **ensemble of 3 models**:

    ---
    **1. 🔤 Naive Bayes (NLP)**
    - Trained on labeled UPI messages
    - Uses Laplace-smoothed word probabilities
    - P(fraud | words) ∝ P(fraud) × ∏ P(word | class)

    ---
    **2. 🌲 Decision Tree**
    - Operates on engineered numerical features
    - Applies learned threshold rules
    - Transparent, explainable predictions

    ---
    **3. 📏 Rule-Based Scorer**
    - Keyword pattern matching
    - Regex for URLs, UPI IDs, phone numbers
    - Domain expert knowledge encoded

    ---
    **Ensemble Weight:**
    - NB: 40%, DT: 30%, Rules: 30%

    ---
    **Features Engineered:**
    - TF-IDF representation
    - Urgency score
    - Reward/threat signals
    - URL & contact pattern detection
    - Lexical statistics
    """)

    st.markdown("---")
    st.markdown("### 🧪 Test Examples")
    examples = {
        "💰 Lottery Scam": "Congratulations! You won Rs 5 lakh! Click http://bit.ly/claim123 to collect your prize now!",
        "🔐 KYC Fraud": "URGENT: Your KYC expired. Verify immediately at tinyurl.com/kyc or your account will be blocked today!",
        "📱 OTP Scam": "Your SBI account has suspicious activity. Share OTP 847291 to verify your identity immediately.",
        "✅ Legit Payment": "Payment of Rs 500 received from Rahul Kumar for grocery bill. Transaction ID: TXN2024001",
        "✅ EMI Reminder": "Reminder: Your home loan EMI of Rs 8500 is due on 15th. Please maintain sufficient balance.",
    }
    selected = st.selectbox("Load an example:", [""] + list(examples.keys()))

# ─── MAIN INPUT ──────────────────────────────────────────────────────────────
default_text = examples.get(selected, "") if selected else ""

col_main, col_info = st.columns([3, 2])

with col_main:
    st.markdown("#### 📨 Enter UPI Message or Transaction Details")
    msg = st.text_area(
        "",
        value=default_text,
        height=130,
        placeholder="Paste a UPI message, SMS, WhatsApp text, or transaction description here...",
        label_visibility="collapsed",
    )

    analyze_btn = st.button("🔍 Analyze with AI/ML", type="primary", use_container_width=True)

with col_info:
    st.markdown("#### ℹ️ How It Works")
    st.markdown("""
    1. **Tokenize** message into word features
    2. **Naive Bayes** scores using word probabilities from training data
    3. **Feature Engineering** extracts 15+ signals (urgency, URL, OTP, etc.)
    4. **Decision Tree** applies learned rules on features
    5. **Ensemble** combines all models with weighted average
    6. **Explainability** shows which features drove the decision
    """)

# ─── ANALYSIS ────────────────────────────────────────────────────────────────
if analyze_btn:
    if not msg.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        result = detect_fraud(msg)
        level    = result["level"]
        scores   = result["model_scores"]
        features = result["features"]

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # ── Risk verdict ──────────────────────────────────────────────────
        css_class = f"risk-{level}"
        icon = {"high": "🚨", "medium": "⚠️", "low": "✅"}[level]
        color_map = {"high": "#ff4444", "medium": "#ffa000", "low": "#43a047"}
        bar_color = color_map[level]

        st.markdown(f"""
        <div class="{css_class}">
          <h2 style="margin:0">{result['status']}</h2>
          <div class="bar-container" style="margin-top:10px">
            <div class="bar-fill" style="width:{result['risk_score']}%; background:{bar_color}"></div>
          </div>
          <p style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; margin:6px 0 0">
            Ensemble Risk Score: <strong>{result['risk_score']}%</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Individual model scores ───────────────────────────────────────
        st.markdown("### 🧠 Model Breakdown (Ensemble Components)")
        m1, m2, m3, m4 = st.columns(4)

        def score_color(s):
            if s >= 60: return "#e53935"
            if s >= 30: return "#fb8c00"
            return "#43a047"

        with m1:
            nb_s = scores["naive_bayes"]
            st.markdown(f"""
            <div class="model-card">
              <div class="model-label">🔤 Naive Bayes</div>
              <div class="model-score" style="color:{score_color(nb_s)}">{nb_s}%</div>
              <div class="bar-container"><div class="bar-fill" style="width:{nb_s}%;background:{score_color(nb_s)}"></div></div>
              <div class="model-desc">NLP word probability model</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            dt_s = scores["decision_tree"]
            st.markdown(f"""
            <div class="model-card">
              <div class="model-label">🌲 Decision Tree</div>
              <div class="model-score" style="color:{score_color(dt_s)}">{dt_s}%</div>
              <div class="bar-container"><div class="bar-fill" style="width:{dt_s}%;background:{score_color(dt_s)}"></div></div>
              <div class="model-desc">Feature-based classifier</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            rb_s = scores["rule_based"]
            st.markdown(f"""
            <div class="model-card">
              <div class="model-label">📏 Rule-Based</div>
              <div class="model-score" style="color:{score_color(rb_s)}">{rb_s}%</div>
              <div class="bar-container"><div class="bar-fill" style="width:{rb_s}%;background:{score_color(rb_s)}"></div></div>
              <div class="model-desc">Expert pattern rules</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            en_s = scores["ensemble"]
            st.markdown(f"""
            <div class="ensemble-badge">
              🎯 {en_s}%
              <div style="font-size:0.75rem;font-weight:300;opacity:0.8;margin-top:4px">Weighted Ensemble</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Explainability ────────────────────────────────────────────────
        col_why, col_feat = st.columns(2)

        with col_why:
            st.markdown("### 🔎 Why This Score?")
            if result["reasons"]:
                for r in result["reasons"]:
                    st.markdown(f'<span class="feature-pill">⚡ {r}</span>', unsafe_allow_html=True)
            else:
                st.success("No suspicious patterns detected.")

        with col_feat:
            st.markdown("### 📐 Engineered Features")
            feat_display = {
                "Word Count": features["word_count"],
                "Urgency Score": features["urgency_score"],
                "Reward Signals": features["reward_score"],
                "Threat Signals": features["threat_score"],
                "Has URL": "Yes" if features["has_url"] else "No",
                "Has Short URL": "Yes" if features["has_short_url"] else "No",
                "Has OTP Request": "Yes" if features["has_otp"] else "No",
                "Has Phone Number": "Yes" if features["has_phone"] else "No",
                "Has UPI ID": "Yes" if features["has_upi"] else "No",
                "Uppercase Ratio": f"{features['uppercase_ratio']:.1%}",
            }
            for k, v in feat_display.items():
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"<small>{k}</small>", unsafe_allow_html=True)
                c2.markdown(f"**{v}**")

        st.markdown("---")

        # ── Safety tips ───────────────────────────────────────────────────
        st.markdown("### 🛡️ Safety Guidelines")
        tip1, tip2, tip3 = st.columns(3)
        with tip1:
            st.info("🔒 **Never share OTP**\nNo bank or UPI app will ever ask for your OTP over message or call.")
        with tip2:
            st.info("🔗 **Avoid unknown links**\nDo not tap shortened URLs (bit.ly, tinyurl) from unknown sources.")
        with tip3:
            st.info("💸 **Verify before paying**\nAlways double-check the UPI ID before confirming any payment.")

        st.markdown("---")
        st.markdown("📞 **Report Cyber Fraud:** Helpline **1930** | [cybercrime.gov.in](https://cybercrime.gov.in)")