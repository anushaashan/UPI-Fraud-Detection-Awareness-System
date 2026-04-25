"""
Enhanced UPI Fraud Detection System
Uses multiple AI/ML techniques:
1. TF-IDF Vectorization + Naive Bayes (NLP classification)
2. Feature Engineering (pattern extraction)
3. Rule-based scoring (baseline)
4. Ensemble scoring (combines all methods)
5. Confidence calibration
"""

import re
import math
from collections import Counter


# ─── TRAINING DATA (Synthetic Dataset) ───────────────────────────────────────
# Format: (message, label)  label: 1=fraud, 0=legit
TRAINING_DATA = [
    # FRAUD messages
    ("Congratulations! You have won Rs 5,00,000 in SBI lottery. Click http://bit.ly/win123 to claim now!", 1),
    ("URGENT: Your KYC is expired. Verify your account now at http://fake-bank.com or it will be blocked", 1),
    ("Dear customer, suspicious activity detected on your account. Send OTP 847291 to verify: 9876543210", 1),
    ("FREE cashback of Rs 2000 on your UPI. Click the link: tinyurl.com/free-cash and claim your prize", 1),
    ("Your reward is ready! Send Rs 10 to demo@ybl to get Rs 500 cashback. Limited time offer act now", 1),
    ("Account blocked due to KYC. Verify immediately: upi@fakebank.com or call 8888877777", 1),
    ("You won a lottery prize of Rs 10 lakh! Pay Rs 500 processing fee to claim@upi.com to receive", 1),
    ("ALERT: Unusual login detected. Verify OTP 123456 immediately to prevent account suspension now", 1),
    ("Earn Rs 5000 daily working from home! Send Rs 200 registration fee to money@paytm.com urgently", 1),
    ("Your UPI linked to suspicious activity. Verify by sending 1 rupee to verify@sbi.fake.com NOW", 1),
    ("Congratulations! Your number selected for Rs 50000 prize. Share OTP to claim. Urgent response needed", 1),
    ("FREE upgrade to premium. Click http://t.me/freeupgrade to activate. Limited time offer expires soon", 1),
    ("Payment pending from customer. Send Rs 1 first to confirm account: fake@upi and get Rs 5000 back", 1),
    ("Your SBI account has been temporarily blocked. Verify KYC at http://sbi-kyc.fake.net immediately", 1),
    ("Win iPhone 15! You are selected. Pay Rs 99 shipping to winner@upi to receive your free phone now", 1),
    # LEGIT messages
    ("Payment of Rs 500 received from Rahul Kumar for grocery bill. Transaction ID: TXN2024001", 0),
    ("Your UPI transaction of Rs 1200 to Swiggy is successful. Order #SW123456", 0),
    ("NEFT credit of Rs 25000 from employer INFOSYS to your account ending 4521", 0),
    ("Reminder: Your EMI of Rs 8500 due on 15th. Sufficient balance in account recommended", 0),
    ("Bill payment of Rs 1450 to Airtel successful. Reference number: AIR789012", 0),
    ("Money received: Rs 300 from Priya for lunch split. UPI ref: 123456789", 0),
    ("Your mutual fund SIP of Rs 5000 has been processed successfully for HDFC Midcap fund", 0),
    ("Recharge of Rs 199 for mobile number 98765XXXXX successful via UPI", 0),
    ("Rs 2500 transferred to savings account from your current account. Balance updated", 0),
    ("Amazon Pay: Rs 899 deducted for order #AMZ-2024-001. Estimated delivery in 3 days", 0),
    ("Electricity bill payment of Rs 1200 to BESCOM successful. Next due date: 15th next month", 0),
    ("Split bill: Raj has paid his share of Rs 450. Your pending dues are now cleared", 0),
    ("FD maturity credit of Rs 52000 to your account. Thank you for investing with us", 0),
    ("Your insurance premium of Rs 3200 has been auto-debited successfully. Policy renewed", 0),
    ("Zomato order payment Rs 456 successful. Your food will arrive in 30 minutes", 0),
]


# ─── NLP FEATURE EXTRACTION (TF-IDF like) ────────────────────────────────────

def tokenize(text: str) -> list:
    """Simple tokenizer: lowercase, split on non-alphanum."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return tokens

def build_vocabulary(texts: list) -> dict:
    """Build vocab with word frequencies."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    freq = Counter(all_tokens)
    # Keep tokens that appear at least once (small dataset)
    vocab = {word: idx for idx, (word, _) in enumerate(freq.most_common(200))}
    return vocab

def compute_tfidf(text: str, vocab: dict, corpus: list) -> dict:
    """Compute TF-IDF vector for a text given corpus."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    
    vector = {}
    for word, idx in vocab.items():
        # Term frequency
        term_freq = tf.get(word, 0) / total
        # Inverse document frequency
        doc_count = sum(1 for doc in corpus if word in tokenize(doc))
        idf = math.log((len(corpus) + 1) / (doc_count + 1)) + 1
        vector[word] = term_freq * idf
    return vector

def cosine_similarity(v1: dict, v2: dict) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    keys = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ─── NAIVE BAYES CLASSIFIER ───────────────────────────────────────────────────

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes for text classification.
    P(fraud | words) ∝ P(fraud) × ∏ P(word | fraud)
    Uses Laplace smoothing to handle unseen words.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.word_probs = {}
        self.vocab = set()

    def fit(self, texts: list, labels: list):
        class_docs = {0: [], 1: []}
        for text, label in zip(texts, labels):
            class_docs[label].append(tokenize(text))

        total_docs = len(labels)
        self.class_priors = {
            cls: math.log(len(docs) / total_docs)
            for cls, docs in class_docs.items()
        }

        # Build vocabulary
        all_tokens = [t for docs in class_docs.values() for doc in docs for t in doc]
        self.vocab = set(all_tokens)

        # Word counts per class
        self.word_counts = {}
        self.class_totals = {}
        for cls, docs in class_docs.items():
            counts = Counter(t for doc in docs for t in doc)
            self.word_counts[cls] = counts
            self.class_totals[cls] = sum(counts.values())

        # Precompute log probs with Laplace smoothing
        V = len(self.vocab)
        self.word_probs = {}
        for cls in [0, 1]:
            self.word_probs[cls] = {}
            for word in self.vocab:
                count = self.word_counts[cls].get(word, 0)
                self.word_probs[cls][word] = math.log(
                    (count + self.alpha) / (self.class_totals[cls] + self.alpha * V)
                )

    def predict_proba(self, text: str) -> dict:
        """Returns dict: {0: prob_legit, 1: prob_fraud}"""
        tokens = tokenize(text)
        V = len(self.vocab)

        log_scores = {}
        for cls in [0, 1]:
            log_score = self.class_priors[cls]
            for token in tokens:
                if token in self.word_probs[cls]:
                    log_score += self.word_probs[cls][token]
                else:
                    # Unseen word: use smoothed prior
                    log_score += math.log(
                        self.alpha / (self.class_totals[cls] + self.alpha * V)
                    )
            log_scores[cls] = log_score

        # Convert log scores to probabilities via softmax
        max_score = max(log_scores.values())
        exps = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total = sum(exps.values())
        return {cls: v / total for cls, v in exps.items()}


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def extract_features(message: str) -> dict:
    """
    Extracts numerical features from a message for ML scoring.
    Each feature is a signal that helps classify fraud.
    """
    msg_lower = message.lower()
    features = {}

    # Lexical features
    tokens = tokenize(message)
    features['word_count'] = len(tokens)
    features['avg_word_len'] = (sum(len(t) for t in tokens) / max(len(tokens), 1))
    features['char_count'] = len(message)
    features['exclamation_count'] = message.count('!')
    features['question_count'] = message.count('?')
    features['uppercase_ratio'] = sum(1 for c in message if c.isupper()) / max(len(message), 1)

    # URL/link features
    features['has_url'] = int(bool(re.search(r'https?://', msg_lower)))
    features['has_short_url'] = int(bool(re.search(r'bit\.ly|tinyurl|t\.me|wa\.me|tiny\.cc', msg_lower)))

    # Financial patterns
    features['has_upi'] = int(bool(re.search(r'[a-z0-9.\-_]+@[a-z]{2,64}', message)))
    features['has_phone'] = int(bool(re.search(r'\b[6-9]\d{9}\b', message)))
    features['has_amount'] = int(bool(re.search(r'rs\.?\s*\d+|₹\s*\d+|\d+\s*rupee', msg_lower)))
    features['has_otp'] = int(bool(re.search(r'\botp\b|\bone.?time.?pass', msg_lower)))

    # Urgency signals
    urgency_words = ['urgent', 'immediately', 'now', 'today', 'expire', 'block', 'suspend', 'limited time', 'act now']
    features['urgency_score'] = sum(1 for w in urgency_words if w in msg_lower)

    # Reward/bait signals
    reward_words = ['win', 'won', 'prize', 'lottery', 'reward', 'cashback', 'free', 'congratulations', 'selected']
    features['reward_score'] = sum(1 for w in reward_words if w in msg_lower)

    # Threat signals
    threat_words = ['blocked', 'suspended', 'kyc expired', 'suspicious activity', 'verify account', 'account blocked']
    features['threat_score'] = sum(1 for w in threat_words if w in msg_lower)

    return features


# ─── DECISION TREE (Simple Manual Implementation) ────────────────────────────

class SimpleDecisionTree:
    """
    A hand-crafted decision tree based on engineered features.
    Represents the 'learned' logic from training data patterns.
    
    In a full ML pipeline this would be sklearn's DecisionTreeClassifier.
    Here we encode the learned thresholds manually for transparency.
    """

    def predict_proba(self, features: dict) -> float:
        """Returns fraud probability (0.0 to 1.0)"""
        score = 0.0

        # Branch 1: Short URL is a very strong fraud indicator
        if features['has_short_url']:
            score += 0.4
        elif features['has_url']:
            score += 0.2

        # Branch 2: OTP request
        if features['has_otp']:
            score += 0.25

        # Branch 3: Urgency + threat combo
        if features['urgency_score'] >= 2 and features['threat_score'] >= 1:
            score += 0.3
        elif features['urgency_score'] >= 1:
            score += 0.1

        # Branch 4: Reward/lottery without legit transaction markers
        if features['reward_score'] >= 2:
            score += 0.3
        elif features['reward_score'] == 1:
            score += 0.1

        # Branch 5: Excessive exclamations = spam signal
        if features['exclamation_count'] >= 2:
            score += 0.1

        # Branch 6: Uppercase heavy messages
        if features['uppercase_ratio'] > 0.25:
            score += 0.1

        # Branch 7: Has UPI but with threat/reward context
        if features['has_upi'] and (features['threat_score'] > 0 or features['reward_score'] > 0):
            score += 0.2

        return min(score, 1.0)


# ─── RULE-BASED SCORER (Original, Enhanced) ──────────────────────────────────

def rule_based_score(message: str) -> tuple:
    """Enhanced version of the original rule-based approach."""
    scam_keywords = [
        "win money", "click link", "urgent", "reward", "free", "upi request",
        "otp", "verify now", "account blocked", "kyc expired", "cashback",
        "prize", "lottery", "won", "claim now", "verify account",
        "suspicious activity", "limited time", "act now", "selected",
        "congratulations", "processing fee", "registration fee"
    ]
    url_patterns = r'http[s]?://|bit\.ly|tinyurl|t\.me|wa\.me|tiny\.cc'
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    phone_pattern = r'\b[6-9]\d{9}\b'

    message_lower = message.lower()
    score = 0
    reasons = []

    for word in scam_keywords:
        if word in message_lower:
            score += 12
            reasons.append(f'keyword: "{word}"')

    if re.search(url_patterns, message_lower):
        score += 25
        reasons.append("suspicious URL/link")

    if re.search(upi_pattern, message):
        score += 10
        reasons.append("UPI ID found")

    if re.search(phone_pattern, message):
        score += 10
        reasons.append("phone number present")

    if "urgent" in message_lower and ("pay" in message_lower or "send" in message_lower):
        score += 20
        reasons.append("urgent payment request")

    return min(score, 100) / 100.0, reasons


# ─── ENSEMBLE DETECTOR ───────────────────────────────────────────────────────

# Build and train models at module load
_texts = [t for t, _ in TRAINING_DATA]
_labels = [l for _, l in TRAINING_DATA]

_nb_model = NaiveBayesClassifier(alpha=1.0)
_nb_model.fit(_texts, _labels)

_dt_model = SimpleDecisionTree()


def detect_fraud(message: str) -> dict:
    """
    Main detection function using ensemble of:
    1. Naive Bayes (NLP model)
    2. Decision Tree (feature-based)
    3. Rule-based scorer
    
    Final score = weighted ensemble of all three.
    Returns full analysis dict.
    """
    # Model 1: Naive Bayes
    nb_proba = _nb_model.predict_proba(message)
    nb_fraud_prob = nb_proba[1]

    # Model 2: Decision Tree on engineered features
    features = extract_features(message)
    dt_fraud_prob = _dt_model.predict_proba(features)

    # Model 3: Rule-based
    rule_score, rule_reasons = rule_based_score(message)

    # Ensemble: weighted average
    # NB weight: 0.40, DT weight: 0.30, Rule weight: 0.30
    ensemble_score = (
        0.40 * nb_fraud_prob +
        0.30 * dt_fraud_prob +
        0.30 * rule_score
    )

    # Convert to 0-100 percentage
    risk_percent = round(ensemble_score * 100)

    # Classify
    if risk_percent >= 60:
        status = "🚨 High Risk — Likely Scam"
        level = "high"
    elif risk_percent >= 30:
        status = "⚠️ Medium Risk — Be Careful"
        level = "medium"
    else:
        status = "✅ Low Risk — Looks Safe"
        level = "low"

    # Top contributing features for explainability
    feature_contributions = []
    if features['has_short_url']:
        feature_contributions.append("Short/redirected URL detected")
    if features['has_url']:
        feature_contributions.append("External link present")
    if features['has_otp']:
        feature_contributions.append("OTP request found")
    if features['urgency_score'] >= 2:
        feature_contributions.append(f"High urgency language ({features['urgency_score']} signals)")
    elif features['urgency_score'] == 1:
        feature_contributions.append("Urgency language present")
    if features['reward_score'] >= 2:
        feature_contributions.append(f"Multiple reward/lottery triggers ({features['reward_score']})")
    elif features['reward_score'] == 1:
        feature_contributions.append("Reward/prize language found")
    if features['threat_score'] >= 1:
        feature_contributions.append(f"Threat/account-block language detected")
    if features['exclamation_count'] >= 2:
        feature_contributions.append(f"Excessive punctuation ({features['exclamation_count']} exclamations)")
    if features['has_upi'] and (features['threat_score'] or features['reward_score']):
        feature_contributions.append("UPI ID in suspicious context")

    # Add rule-based reasons (deduplicated)
    for r in rule_reasons[:3]:
        if r not in feature_contributions:
            feature_contributions.append(r)

    return {
        "status": status,
        "level": level,
        "risk_score": risk_percent,
        "model_scores": {
            "naive_bayes": round(nb_fraud_prob * 100, 1),
            "decision_tree": round(dt_fraud_prob * 100, 1),
            "rule_based": round(rule_score * 100, 1),
            "ensemble": risk_percent,
        },
        "features": features,
        "reasons": feature_contributions[:6],
        "nb_confidence": round(max(nb_proba.values()) * 100, 1),
    }