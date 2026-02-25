import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="TrustNet", page_icon="🛡", layout="wide")

st.title("🛡 TrustNet")
st.subheader("Advanced AI Fake Account & Scam Detection System")
st.markdown("---")

# ---------------- TRAIN SIMPLE ML MODEL ----------------
# Dummy dataset (for demo ML model)
data = pd.DataFrame({
    "followers": [10, 500, 30, 1000, 20, 800],
    "following": [300, 400, 200, 900, 500, 700],
    "posts": [1, 100, 2, 300, 3, 200],
    "account_age": [5, 365, 10, 400, 15, 500],
    "fake": [1, 0, 1, 0, 1, 0]
})

X = data[["followers", "following", "posts", "account_age"]]
y = data["fake"]

model = LogisticRegression()
model.fit(X, y)

# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Choose Module",
    ("Fake Account Detector", "Bulk CSV Analysis", "Scam Message Detector")
)

# =====================================================
# 1️⃣ FAKE ACCOUNT DETECTOR (ML VERSION)
# =====================================================
if option == "Fake Account Detector":
    st.header("🔍 AI Fake Account Detection")

    col1, col2 = st.columns(2)

    with col1:
        followers = st.number_input("Followers", min_value=0)
        following = st.number_input("Following", min_value=0)
        posts = st.number_input("Posts", min_value=0)

    with col2:
        account_age = st.number_input("Account Age (days)", min_value=0)
        username = st.text_input("Username")

    if st.button("Analyze Account"):

        input_data = np.array([[followers, following, posts, account_age]])
        probability = model.predict_proba(input_data)[0][1]
        risk_score = int(probability * 100)

        st.subheader("📊 Risk Score")
        st.progress(risk_score)
        st.metric("Fake Probability", f"{risk_score}%")

        # Username risk check
        suspicious_patterns = ["123", "official", "crypto", "free", "win"]
        username_risk = any(word in username.lower() for word in suspicious_patterns)

        if risk_score > 70 or username_risk:
            st.error("🚨 High Risk Fake Profile")
        elif risk_score > 40:
            st.warning("⚠ Medium Risk Profile")
        else:
            st.success("✅ Low Risk Profile")

        # Visualization
        fig, ax = plt.subplots()
        ax.bar(["Risk Score"], [risk_score])
        ax.set_ylim(0, 100)
        st.pyplot(fig)

# =====================================================
# 2️⃣ BULK CSV ANALYSIS
# =====================================================
elif option == "Bulk CSV Analysis":
    st.header("📂 Bulk Fake Account Analysis")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview Data")
        st.dataframe(df.head())

        if st.button("Run Bulk Analysis"):
            X_bulk = df[["followers", "following", "posts", "account_age"]]
            probs = model.predict_proba(X_bulk)[:, 1]
            df["Fake Probability (%)"] = (probs * 100).astype(int)

            st.success("Analysis Complete")
            st.dataframe(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "TrustNet_Results.csv",
                "text/csv"
            )

# =====================================================
# 3️⃣ SCAM MESSAGE DETECTOR
# =====================================================
elif option == "Scam Message Detector":
    st.header("📩 AI Scam Message Detection")

    message = st.text_area("Paste Message Here")

    if st.button("Analyze Message"):

        scam_keywords = ["invest", "profit", "otp", "urgent", "click", "prize", "bitcoin"]
        risk_score = sum(word in message.lower() for word in scam_keywords) * 15
        risk_score = min(risk_score, 100)

        st.progress(risk_score)
        st.metric("Scam Probability", f"{risk_score}%")

        if risk_score > 50:
            st.error("🚨 High Scam Probability")
        elif risk_score > 20:
            st.warning("⚠ Medium Scam Probability")
        else:
            st.success("✅ Low Scam Probability")