# 🛡️ UPI Fraud Detection System

An AI/ML-powered system to detect fraudulent UPI messages using an **ensemble of machine learning models** including Naive Bayes, Decision Tree, and Rule-Based scoring.

---

## 🤖 AI/ML Techniques Used

| Technique | Description |
|---|---|
| **Naive Bayes Classifier** | NLP-based probabilistic model trained on labeled UPI messages using Laplace smoothing |
| **Decision Tree** | Feature-based classifier using 15+ engineered signals |
| **TF-IDF Vectorization** | Term Frequency–Inverse Document Frequency for text representation |
| **Feature Engineering** | Extracts urgency score, reward signals, URL patterns, lexical stats |
| **Ensemble Learning** | Weighted combination of all 3 models (NB: 40%, DT: 30%, Rules: 30%) |
| **Explainable AI (XAI)** | Shows which features contributed to each prediction |

---

## 📁 Project Structure

```
upi-fraud-detector/
├── app.py                  # Streamlit web application
├── fraud_detection.py      # ML engine (Naive Bayes + Decision Tree + Rules)
└── README.md
```

---

## ⚙️ How It Works

1. **Input** — User pastes a UPI message or SMS
2. **Tokenization** — Message is broken into word tokens
3. **Naive Bayes** — Computes P(fraud | words) using trained word probabilities
4. **Feature Extraction** — 15+ numerical features extracted (urgency, OTP, URLs, etc.)
5. **Decision Tree** — Applies learned threshold rules on engineered features
6. **Ensemble Score** — Weighted average of all 3 model outputs
7. **Explainability** — Displays which features drove the final decision

---

## 🚀 Installation & Running

```bash
# Install dependencies
pip install streamlit

# Run the app
streamlit run app.py
```

---

## 📊 Features Extracted

- Word count, average word length, uppercase ratio
- Urgency score (urgent, immediately, expire, block...)
- Reward/lottery signals (win, prize, cashback, free...)
- Threat signals (account blocked, KYC expired...)
- URL detection (http, bit.ly, tinyurl, t.me...)
- UPI ID and phone number presence
- OTP request detection
- Exclamation/punctuation count

---

## 🧠 Model Details

### Naive Bayes
Uses Multinomial Naive Bayes with Laplace (add-1) smoothing:

```
P(fraud | message) ∝ P(fraud) × ∏ P(word | fraud)
```

Trained on 30 labeled UPI messages (15 fraud, 15 legitimate).

### Decision Tree
Hand-crafted decision rules learned from feature patterns:
- Short URL → +40% fraud probability
- OTP request → +25%
- Urgency + threat combo → +30%
- Reward/lottery signals → +30%

### Ensemble
```
Final Score = 0.40 × NaiveBayes + 0.30 × DecisionTree + 0.30 × RuleBased
```

---

## 🛡️ Safety Tips

- **Never share your OTP** with anyone
- **Avoid clicking** shortened or unknown links
- **Verify UPI ID** before making any payment
- **Report fraud** to Helpline: **1930** or [cybercrime.gov.in](https://cybercrime.gov.in)

---

## 📚 Topics Covered (AI/ML Fundamentals)

- Text Classification
- Natural Language Processing (NLP)
- Probabilistic Models (Naive Bayes)
- Decision Trees
- Feature Engineering
- TF-IDF Representation
- Ensemble Methods
- Explainable AI
