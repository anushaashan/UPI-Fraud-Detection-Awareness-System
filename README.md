# UPI-Fraud-Detection-Awareness-System
A Python-based web application that helps users identify potentially fraudulent UPI messages and links using keyword-based risk scoring.

📌 Problem Statement
UPI (Unified Payments Interface) scams are increasingly common in India. Users often receive fake payment requests, phishing links, and urgent messages designed to trick them into sharing OTPs or clicking malicious links. This tool helps everyday users quickly check whether a UPI message is safe before acting on it.

💡 Features

Paste any UPI message or link and get an instant safety verdict
Risk scoring system (0–100%) based on suspicious keyword detection
Classifies messages as ✅ Safe, ⚠️ Medium Risk, or ⚠️ High Risk
Shows the exact reasons why a message was flagged
Built-in Safety Tips and Fraud Reporting information


🛠️ Tech Stack
ToolPurposePythonCore languageStreamlitWeb app frameworkCustom logicKeyword-based fraud detection

📁 Project Structure
upi-fraud-detector/
│
├── app.py                # Streamlit frontend
├── fraud_detection.py    # Fraud detection logic
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

⚙️ Setup & Installation
1. Clone the Repository
bashgit clone https://github.com/your-username/upi-fraud-detector.git
cd upi-fraud-detector
2. Install Dependencies
Make sure you have Python 3.7+ installed, then run:
bashpip install -r requirements.txt
3. Run the App
bashstreamlit run app.py
The app will open in your browser at http://localhost:8501

📦 requirements.txt
streamlit

🚀 How to Use

Open the app in your browser
Paste a UPI message or suspicious link into the text box
Click the Check button
View the result:

✅ Safe — No suspicious content found
⚠️ Medium Risk — One suspicious keyword detected
⚠️ High Risk - Possible Scam — Multiple suspicious keywords detected


Read the flagged reasons and follow the Safety Tips


🧪 Example Test Cases
MessageExpected ResultSending you 200 rupees for lunch✅ SafeClick link to claim your reward⚠️ High RiskUrgent UPI request pending⚠️ High RiskWin money now, free offer!⚠️ High Risk

🔍 How Detection Works
The fraud_detection.py module scans messages for a list of known scam keywords:
"win money", "click link", "urgent", "reward", "free", "upi request"
Each match adds 20 points to a risk score:

Score ≥ 40 → High Risk
Score > 0  → Medium Risk
Score = 0  → Safe


📞 Report Fraud
If you receive a UPI scam:

📞 Call 1930 (National Cybercrime Helpline)
🌐 Visit cybercrime.gov.in


👨‍💻 Author
Made as part of a Python course project to address a real-world digital safety problem in India.
