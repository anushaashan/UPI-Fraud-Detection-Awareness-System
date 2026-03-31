import re

def detect_fraud(message):
    scam_keywords = [
        "win money", "click link", "urgent", "reward",
        "free", "upi request", "otp", "verify now",
        "account blocked", "kyc expired", "cashback",
        "prize", "lottery", "won", "claim now", "verify account",
        "suspicious activity", "limited time", "act now"
    ]

    url_patterns = r'http[s]?://|bit\.ly|tinyurl|t\.me|wa\.me|tiny\.cc'
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    phone_pattern = r'\b[6-9]\d{9}\b'

    message_lower = message.lower()
    score = 0
    reasons = []

    for word in scam_keywords:
        if word in message_lower:
            score += 15
            reasons.append(word)

    if re.search(url_patterns, message_lower):
        score += 25
        reasons.append("suspicious link")

    if re.search(upi_pattern, message):
        score += 10
        reasons.append("upi id found")

    if re.search(phone_pattern, message):
        score += 10
        reasons.append("phone number found")

    if "urgent" in message_lower and ("pay" in message_lower or "send" in message_lower):
        score += 20
        reasons.append("urgent payment request")

    if score > 100:
        score = 100

    if score >= 50:
        status = "High Risk - Likely Scam"
    elif score > 0:
        status = "Medium Risk - Be Careful"
    else:
        status = "Looks Safe"

    return status, score, reasons
