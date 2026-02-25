import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="TrustNet AI Security",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- REMOVE STREAMLIT HEADER ----------------
st.set_page_config(
    page_title="TrustNet AI Security",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------- SPLASH SCREEN ----------------
splash = st.empty()
with splash.container():
    try:
        st.image("logo.png", width=200)
    except:
        pass
    st.title("TrustNet")
    st.write("Initializing AI Security Engine...")
    time.sleep(2)
splash.empty()

# ---------------- OFFLINE WARNING ----------------
st.warning("⚠ If this page does not load, please check your internet connection.")

# ---------------- HEADER ----------------
try:
    st.image("logo.png", width=150)
except:
    pass

st.title("🛡 TrustNet")
st.markdown("""
### 🌍 Protecting Digital Identities Worldwide

TrustNet uses Artificial Intelligence to detect:
- Fake social media accounts  
- Scam messages  
- Fraud risk probability  

Built for safer online communities.
""")
st.markdown("---")

# ---------------- LOAD DATASET ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("fake_accounts_dataset.csv")

    FEATURES = [
        "profile pic",
        "nums/length username",
        "fullname words",
        "nums/length fullname",
        "name==username",
        "description length",
        "external URL",
        "private",
        "#posts",
        "#followers",
        "#follows"
    ]

    X = df[FEATURES]
    y = df["fake"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    return model, accuracy, cm, FEATURES


model, accuracy, cm, FEATURES = load_model()

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio(
    "Choose Module",
    ("Fake Account Detector", "Bulk CSV Analysis", "Scam Message Detector")
)

st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# Confusion Matrix (Matplotlib only)
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax_cm.text(j, i, cm[i][j], ha="center", va="center")

ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.sidebar.pyplot(fig_cm)

# =====================================================
# 1️⃣ FAKE ACCOUNT DETECTOR
# =====================================================
if option == "Fake Account Detector":

    st.header("🔍 AI Fake Account Detection")

    col1, col2 = st.columns(2)

    with col1:
        profile_pic = st.selectbox("Profile Picture", [0, 1])
        username_ratio = st.number_input("Nums/Length Username", min_value=0.0)
        fullname_words = st.number_input("Fullname Words", min_value=0)
        fullname_ratio = st.number_input("Nums/Length Fullname", min_value=0.0)
        name_equal = st.selectbox("Name == Username", [0, 1])
        description_length = st.number_input("Description Length", min_value=0)

    with col2:
        external_url = st.selectbox("External URL", [0, 1])
        private = st.selectbox("Private Account", [0, 1])
        posts = st.number_input("Posts", min_value=0)
        followers = st.number_input("Followers", min_value=0)
        follows = st.number_input("Follows", min_value=0)

    if st.button("Analyze Account"):

        with st.spinner("🔐 Running AI Fraud Detection..."):

            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1)

            input_data = np.array([[ 
                profile_pic,
                username_ratio,
                fullname_words,
                fullname_ratio,
                name_equal,
                description_length,
                external_url,
                private,
                posts,
                followers,
                follows
            ]])

            probability = model.predict_proba(input_data)[0][1]
            risk_score = int(probability * 100)

        st.subheader("📊 Fake Probability Score")
        st.progress(risk_score)
        st.metric("Fake Probability", f"{risk_score}%")

        if risk_score > 70:
            st.error("🚨 High Risk Fake Profile")
        elif risk_score > 40:
            st.warning("⚠ Medium Risk Profile")
        else:
            st.success("✅ Low Risk Profile")

# =====================================================
# 2️⃣ BULK CSV ANALYSIS
# =====================================================
elif option == "Bulk CSV Analysis":

    st.header("📂 Bulk Fake Account Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        bulk_df = pd.read_csv(uploaded_file)
        st.dataframe(bulk_df.head())

        if st.button("Run Bulk Analysis"):

            if all(col in bulk_df.columns for col in FEATURES):

                X_bulk = bulk_df[FEATURES]
                probs = model.predict_proba(X_bulk)[:, 1]
                bulk_df["Fake Probability (%)"] = (probs * 100).astype(int)

                st.success("Analysis Complete")
                st.dataframe(bulk_df)

                st.download_button(
                    "Download Results",
                    bulk_df.to_csv(index=False),
                    "TrustNet_Results.csv",
                    "text/csv"
                )
            else:
                st.error("Uploaded CSV format is incorrect.")

# =====================================================
# 3️⃣ SCAM MESSAGE DETECTOR
# =====================================================
elif option == "Scam Message Detector":

    st.header("📩 AI Scam Message Detection")
    message = st.text_area("Paste Message Here")

    if st.button("Analyze Message"):

        scam_keywords = [
            "invest", "profit", "otp", "urgent",
            "click", "prize", "bitcoin"
        ]

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