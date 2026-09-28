import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="ABC Ltd. Supply Logistics", page_icon="📦", layout="wide")

@st.cache_data
def train_models():
    np.random.seed(42)
    n = 1000
    vol = np.random.randint(200, 3500, n)
    analyzers = np.random.randint(1, 7, n)
    lead_time = np.round(np.random.uniform(1.5, 18.0, n), 1)
    surge = np.random.choice([0, 1], n, p=[0.7, 0.3])
    outbreak = np.round(np.random.uniform(1.0, 5.0, n), 1)

    req_units = (12.0 + 0.045*vol + 5.2*analyzers + 8.5*surge + 6.0*outbreak + np.random.normal(0, 8, n)).astype(int)
    buffer = np.clip(req_units + np.random.randint(-40, 50, n), 5, 260)
    
    shortage = req_units - buffer
    logit = -1.2 + 0.08*shortage + 0.15*lead_time + 0.4*surge
    prob = 1 / (1 + np.exp(-logit))
    risk = (prob > 0.45).astype(int)

    df = pd.DataFrame({
        'vol': vol, 'analyzers': analyzers, 'buffer': buffer, 
        'lead_time': lead_time, 'surge': surge, 'outbreak': outbreak, 
        'req_units': req_units, 'risk': risk
    })

    X_r = df[['vol', 'analyzers', 'surge', 'outbreak']]
    y_r = df['req_units']
    reg = LinearRegression().fit(X_r, y_r)

    X_c = df[['vol', 'buffer', 'lead_time', 'surge', 'outbreak']]
    y_c = df['risk']
    clf = LogisticRegression(max_iter=1000).fit(X_c, y_c)

    return reg, clf

reg, clf = train_models()

st.title("📦 ABC Ltd. Inter-Lab Supply Logistics Tool")
st.markdown("Decision Support Tool for Reagent Inventory & Stockout Risk Management")

st.sidebar.header("Spoke Lab Parameters")
vol_in = st.sidebar.slider("Daily Patient Sample Volume", 200, 3500, 1450)
analyzers_in = st.sidebar.slider("Active Automated Analyzers", 1, 6, 3)
buffer_in = st.sidebar.slider("Current Buffer Stock (Kits)", 10, 250, 65)
lead_in = st.sidebar.slider("Transit Lead Time (Hours)", 1.0, 18.0, 6.5)
surge_in = 1 if st.sidebar.radio("Weekend Surge?", ["No", "Yes"]) == "Yes" else 0
outbreak_in = st.sidebar.slider("Outbreak Index", 1.0, 5.0, 2.1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Demand Prediction")
    pred_kits = reg.predict([[vol_in, analyzers_in, surge_in, outbreak_in]])[0]
    st.metric("Predicted Reagent Demand", f"{int(np.ceil(pred_kits))} Kits")
    st.info(f"Recommended Dispatch: **{max(0, int(np.ceil(pred_kits - buffer_in)))} Kits**")

with col2:
    st.subheader("2. 48-Hour Stockout Risk")
    prob = clf.predict_proba([[vol_in, buffer_in, lead_in, surge_in, outbreak_in]])[0][1]
    if prob >= 0.45:
        st.error(f"🚨 **HIGH RISK**: {prob*100:.1f}% Probability")
    else:
        st.success(f"🟢 **SAFE**: {prob*100:.1f}% Probability")
    st.progress(float(prob))
