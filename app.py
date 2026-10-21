import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Configure page layout
st.set_page_config(
    page_title="Customer Retention & Churn Intelligence",
    page_icon="⚡",
    layout="wide"
)

# Load pipeline and schemas
@st.cache_resource
def load_artifacts():
    return joblib.load("churn_model.joblib")

try:
    artifacts = load_artifacts()
    pipeline = artifacts["pipeline"]
    cat_options = artifacts["cat_options"]
except Exception as e:
    st.error(f"Error loading 'churn_model.joblib': {e}. Please ensure the model artifact is saved.")
    st.stop()

# --- PRE-LOADED CUSTOMER BINDER PROFILES ---
CUSTOMER_BINDER = {
    "Select a customer profile...": None,
    "📌 Cust #9237 - Sarah L. (High Risk Preset)": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70, "TotalCharges": 151.65
    },
    "📌 Cust #3668 - David K. (Mid Risk Preset)": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 18, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 53.85, "TotalCharges": 969.30
    },
    "📌 Cust #7795 - Elena R. (Low Risk Preset)": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 58, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 42.30, "TotalCharges": 2453.40
    }
}

# --- HEADER ---
st.title("⚡ Customer Churn Diagnostics & Intelligence")
st.caption("Identify churn vulnerabilities, test retention incentives, and batch-score customer portfolios.")

# Navigation tabs
tab_individual, tab_batch = st.tabs(["🎯 Customer Profiler & Simulator", "📁 Batch Customer Scanner"])

# =========================================================
# TAB 1: INDIVIDUAL PROFILER & WHAT-IF SIMULATOR
# =========================================================
with tab_individual:
    # 1. Customer Binder / Presets Section
    st.markdown("### 🗂️ Sample Customer Binder")
    selected_preset = st.selectbox(
        "Choose a pre-configured customer profile to auto-populate the dashboard:",
        options=list(CUSTOMER_BINDER.keys()),
        index=1  # Default to High Risk preset for instant demonstration
    )

    preset_data = CUSTOMER_BINDER[selected_preset]
    if preset_data is None:
        # Fallback default values
        preset_data = {
            "gender": "Male", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
            "tenure": 6, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.0, "TotalCharges": 450.0
        }

    st.markdown("---")

    # 2. Layout Form Inputs
    col_left, col_mid, col_right = st.columns([1.1, 1.1, 1.4])

    with col_left:
        st.subheader("📋 Account & Contract")
        contract = st.selectbox("Contract Plan", cat_options.get("Contract", ["Month-to-month", "One year", "Two year"]),
                                index=cat_options.get("Contract", []).index(preset_data["Contract"]) if preset_data["Contract"] in cat_options.get("Contract", []) else 0)
        tenure = st.slider("Tenure (Months active)", min_value=1, max_value=72, value=int(preset_data["tenure"]))
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(preset_data["MonthlyCharges"]), step=0.5)
        total_charges = float(tenure * monthly_charges)
        st.caption(f"Estimated Total Revenue: **${total_charges:,.2f}**")

        paperless = st.radio("Paperless Billing", cat_options.get("PaperlessBilling", ["Yes", "No"]),
                             index=cat_options.get("PaperlessBilling", []).index(preset_data["PaperlessBilling"]) if preset_data["PaperlessBilling"] in cat_options.get("PaperlessBilling", []) else 0, horizontal=True)
        payment_method = st.selectbox("Payment Method", cat_options.get("PaymentMethod", []),
                                      index=cat_options.get("PaymentMethod", []).index(preset_data["PaymentMethod"]) if preset_data["PaymentMethod"] in cat_options.get("PaymentMethod", []) else 0)

    with col_mid:
        st.subheader("🛠️ Subscribed Services")
        internet_service = st.selectbox("Internet Technology", cat_options.get("InternetService", []),
                                        index=cat_options.get("InternetService", []).index(preset_data["InternetService"]) if preset_data["InternetService"] in cat_options.get("InternetService", []) else 0)
        
        tech_support = st.selectbox("Tech Support", cat_options.get("TechSupport", []),
                                    index=cat_options.get("TechSupport", []).index(preset_data["TechSupport"]) if preset_data["TechSupport"] in cat_options.get("TechSupport", []) else 0)
        
        online_security = st.selectbox("Online Security", cat_options.get("OnlineSecurity", []),
                                       index=cat_options.get("OnlineSecurity", []).index(preset_data["OnlineSecurity"]) if preset_data["OnlineSecurity"] in cat_options.get("OnlineSecurity", []) else 0)

        with st.expander("Additional Services & Demographics"):
            phone_service = st.selectbox("Phone Service", cat_options.get("PhoneService", []),
                                         index=cat_options.get("PhoneService", []).index(preset_data["PhoneService"]) if preset_data["PhoneService"] in cat_options.get("PhoneService", []) else 0)
            multiple_lines = st.selectbox("Multiple Lines", cat_options.get("MultipleLines", []),
                                          index=cat_options.get("MultipleLines", []).index(preset_data["MultipleLines"]) if preset_data["MultipleLines"] in cat_options.get("MultipleLines", []) else 0)
            online_backup = st.selectbox("Online Backup", cat_options.get("OnlineBackup", []),
                                         index=cat_options.get("OnlineBackup", []).index(preset_data["OnlineBackup"]) if preset_data["OnlineBackup"] in cat_options.get("OnlineBackup", []) else 0)
            device_protection = st.selectbox("Device Protection", cat_options.get("DeviceProtection", []),
                                             index=cat_options.get("DeviceProtection", []).index(preset_data["DeviceProtection"]) if preset_data["DeviceProtection"] in cat_options.get("DeviceProtection", []) else 0)
            streaming_tv = st.selectbox("Streaming TV", cat_options.get("StreamingTV", []),
                                        index=cat_options.get("StreamingTV", []).index(preset_data["StreamingTV"]) if preset_data["StreamingTV"] in cat_options.get("StreamingTV", []) else 0)
            streaming_movies = st.selectbox("Streaming Movies", cat_options.get("StreamingMovies", []),
                                            index=cat_options.get("StreamingMovies", []).index(preset_data["StreamingMovies"]) if preset_data["StreamingMovies"] in cat_options.get("StreamingMovies", []) else 0)
            gender = st.selectbox("Gender", cat_options.get("gender", []),
                                  index=cat_options.get("gender", []).index(preset_data["gender"]) if preset_data["gender"] in cat_options.get("gender", []) else 0)
            senior = st.selectbox("Senior Citizen", cat_options.get("SeniorCitizen", []),
                                  index=cat_options.get("SeniorCitizen", []).index(preset_data["SeniorCitizen"]) if preset_data["SeniorCitizen"] in cat_options.get("SeniorCitizen", []) else 0)
            partner = st.selectbox("Partner", cat_options.get("Partner", []),
                                   index=cat_options.get("Partner", []).index(preset_data["Partner"]) if preset_data["Partner"] in cat_options.get("Partner", []) else 0)
            dependents = st.selectbox("Dependents", cat_options.get("Dependents", []),
                                      index=cat_options.get("Dependents", []).index(preset_data["Dependents"]) if preset_data["Dependents"] in cat_options.get("Dependents", []) else 0)

    # 3. Assemble Record and Run Inference Live
    customer_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    churn_prob = pipeline.predict_proba(customer_input)[0][1]

    # Categorize Risk
    if churn_prob >= 0.60:
        risk_level = "High Risk"
        risk_color = "#FF4B4B"
        action_advice = "🚨 **Immediate Retention Action Required:** Offer an upgraded multi-month contract with a dedicated discount or add free Tech Support/Security bundle."
    elif churn_prob >= 0.30:
        risk_level = "Medium Risk"
        risk_color = "#FFAA00"
        action_advice = "⚠️ **Monitoring Recommended:** Customer shows signs of churn sensitivity. Target with a loyalty reward or annual contract renewal nudge."
    else:
        risk_level = "Low Risk"
        risk_color = "#00CC96"
        action_advice = "✅ **Healthy Relationship:** Low probability of departure. Prime candidate for cross-selling premium add-on services."

    with col_right:
        st.subheader("📊 Live Risk Score")
        
        # Interactive Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_prob * 100, 1),
            number={'suffix': "%", 'font': {'size': 36}},
            title={'text': f"Risk Category: <b>{risk_level}</b>", 'font': {'size': 20, 'color': risk_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e0e0e0",
                'steps': [
                    {'range': [0, 30], 'color': '#D9F7E8'},
                    {'range': [30, 60], 'color': '#FFF3CD'},
                    {'range': [60, 100], 'color': '#FCE8E6'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': round(churn_prob * 100, 1)
                }
            }
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.info(action_advice)

# =========================================================
# TAB 2: BATCH CUSTOMER SCANNER (CSV UPLOAD)
# =========================================================
with tab_batch:
    st.subheader("📁 Upload Customer Portfolio")
    st.markdown("Upload a raw CSV file containing customer records to score all accounts simultaneously.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Clean potential empty strings in TotalCharges
        if "TotalCharges" in batch_df.columns:
            batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce").fillna(0)
        if "SeniorCitizen" in batch_df.columns:
            batch_df["SeniorCitizen"] = batch_df["SeniorCitizen"].replace({0: "No", 1: "Yes"})

        try:
            # Predict batch
            probs = pipeline.predict_proba(batch_df)[:, 1]
            batch_df["Churn_Probability"] = np.round(probs * 100, 1)
            batch_df["Risk_Level"] = pd.cut(
                batch_df["Churn_Probability"],
                bins=[-1, 30, 60, 100],
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )

            # High-level metrics
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Customers", len(batch_df))
            b_col2.metric("High Risk Tiers", int((batch_df["Risk_Level"] == "High Risk").sum()))
            b_col3.metric("Medium Risk Tiers", int((batch_df["Risk_Level"] == "Medium Risk").sum()))
            b_col4.metric("Low Risk Tiers", int((batch_df["Risk_Level"] == "Low Risk").sum()))

            # Display table
            st.dataframe(
                batch_df[["tenure", "Contract", "MonthlyCharges", "PaymentMethod", "Churn_Probability", "Risk_Level"]].head(25),
                use_container_width=True
            )

            # Download evaluated records
            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored Portfolio as CSV",
                data=csv_export,
                file_name="churn_risk_portfolio_evaluated.csv",
                mime="text/csv"
            )
        except Exception as err:
            st.error(f"Error evaluating batch dataset: {err}. Please ensure column names match the training data.")
    else:
        st.caption("Upload a `.csv` with columns matching the Telco churn dataset to start automated scanning.")