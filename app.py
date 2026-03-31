import streamlit as st
from fraud_detection import detect_fraud

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🔐")

st.title("UPI Fraud Detection System")

st.write("Enter any UPI message or link to check if it is safe or not.")

msg = st.text_area("Enter message or link")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Please enter a message first")
    else:
        status, score, reasons = detect_fraud(msg)

        st.subheader("Result")
        st.write(status)
        st.write("Risk Score:", score, "%")

        if score >= 50:
            st.error("High risk message")
        elif score > 0:
            st.warning("Be careful")
        else:
            st.success("Looks safe")

        if reasons:
            st.subheader("Reason")
            for r in reasons:
                st.write("-", r)

        st.subheader("Safety Tips")
        st.write("""
        Never share OTP  
        Do not click unknown links  
        Always verify before payment  
        Avoid urgent messages  
        Use trusted apps only  
        """)

        st.subheader("Report Fraud")
        st.write("Helpline: 1930")
        st.write("cybercrime.gov.in")
