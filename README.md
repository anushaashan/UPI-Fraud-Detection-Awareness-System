#  UPI Fraud Detection & Awareness System

A Python-based web application built using **Streamlit** that helps users detect suspicious UPI messages, payment links, and scam texts commonly found in India.

---

##  Project Overview

UPI fraud is one of the most common cybercrimes in India. This project aims to create a simple and effective tool that:

- Analyzes UPI messages for suspicious patterns
- Detects scam keywords, fake links, and urgent payment requests
- Gives a risk score from 0% to 100%
- Educates users with safety tips
- Provides fraud reporting resources (Cybercrime Helpline: 1930)

---

##  Project Structure

```
upi-fraud-detection/
├── app.py               # Main Streamlit web application
├── fraud_detection.py   # Core fraud detection logic
└── README.md            # Project documentation
```

---

##  Requirements

- Python 3.7 or above
- Streamlit library

---

##  How to Run

### Step 1 — Install Python
Make sure Python is installed on your system. You can check by running:
```
python --version
```

### Step 2 — Install Streamlit
Open your terminal or command prompt and run:
```
pip install streamlit
```

### Step 3 — Navigate to the Project Folder
```
cd path/to/upi-fraud-detection
```

### Step 4 — Run the Application
```
streamlit run app.py
```

### Step 5 — Open in Browser
After running the command, Streamlit will automatically open the app in your browser at:
```
http://localhost:8501
```

---

## How It Works

The application is split into two files:

### `fraud_detection.py`
This file contains the `check_fraud(message)` function which:
- Scans the input message for **scam keywords** like "win money", "urgent", "OTP", "KYC expired", etc.
- Uses **Regular Expressions (regex)** to detect:
  - Suspicious URLs and shortened links (e.g., bit.ly, tinyurl)
  - UPI IDs embedded in messages
  - Phone numbers
- Flags **urgent payment requests** as a high-risk combo
- Returns a **risk status**, **score (0–100%)**, and list of **reasons**

### `app.py`
This file is the front-end built with **Streamlit**:
- Takes user input (UPI message or link)
- Calls `check_fraud()` from `fraud_detection.py`
- Displays the result with color-coded alerts (Red / Yellow / Green)
- Shows a visual risk score progress bar
- Lists all flagged reasons
- Provides safety tips and fraud reporting links

---

## Sample Test Cases

| Message | Expected Result |
|---|---|
| "Congratulations! You won Rs 50,000. Click bit.ly/claim now" |  High Risk |
| "URGENT: Your account is blocked. Verify now via OTP" |  High Risk |
| "You have a cashback reward waiting. Contact us." | Medium Risk |
| "Please pay Rs 500 for dinner. My UPI is rahul@okicici" | Safe |

---

## Safety Tips Covered in the App

- Never share your OTP with anyone
- Avoid clicking unknown or shortened links
- Always verify the sender before making a payment
- Do not trust urgent or reward-based messages
- Use only trusted UPI apps like BHIM, GPay, PhonePe, or Paytm

---

## Report Fraud

- **Helpline:** Call **1930** (National Cyber Crime Helpline)
- **Website:** [https://cybercrime.gov.in](https://cybercrime.gov.in)

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Regular Expressions (re) | Pattern detection in messages |


