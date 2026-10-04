import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ============================================================
# Insurance Claim Fraud Detection - Streamlit App
# Models expected in the same folder as this file:
#   random_forest_fraud_model.pkl
#   isolation_forest_fraud_model.pkl
#   isolation_forest_preprocessor.pkl
# ============================================================

st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

RF_PATH = BASE_DIR / "random_forest_fraud_model.pkl"
ISO_PATH = BASE_DIR / "isolation_forest_fraud_model.pkl"
ISO_PREPROCESSOR_PATH = BASE_DIR / "isolation_forest_preprocessor.pkl"


def patch_old_sklearn_imputers(obj, seen=None):
    """
    Compatibility fix for models saved with scikit-learn 1.7.x
    and loaded with scikit-learn 1.8.x.

    scikit-learn 1.8 expects SimpleImputer._fill_dtype.
    Older pickles may not contain this internal attribute.
    """
    from sklearn.impute import SimpleImputer

    if seen is None:
        seen = set()

    if id(obj) in seen:
        return

    seen.add(id(obj))

    if (
        isinstance(obj, SimpleImputer)
        and not hasattr(obj, "_fill_dtype")
        and hasattr(obj, "statistics_")
    ):
        # For fitted imputers, statistics_ has the dtype used during fit.
        obj._fill_dtype = obj.statistics_.dtype

    # Walk through fitted pipelines and column transformers.
    for attr in ("steps", "transformers_", "transformers"):
        items = getattr(obj, attr, None)

        if items:
            for item in items:
                if len(item) >= 2:
                    transformer = item[1]

                    if transformer == "passthrough" or transformer == "drop":
                        continue

                    patch_old_sklearn_imputers(transformer, seen)


@st.cache_resource
def load_models():
    with open(RF_PATH, "rb") as f:
        rf_model = pickle.load(f)

    with open(ISO_PATH, "rb") as f:
        iso_model = pickle.load(f)

    with open(ISO_PREPROCESSOR_PATH, "rb") as f:
        iso_preprocessor = pickle.load(f)

    # Compatibility fix for your models saved with scikit-learn 1.7.2
    # when the Streamlit environment has scikit-learn 1.8.x.
    patch_old_sklearn_imputers(rf_model)
    patch_old_sklearn_imputers(iso_preprocessor)

    return rf_model, iso_model, iso_preprocessor


# Exact 49 features expected by the saved Random Forest model.
MODEL_FEATURES = [
    "claim_amount",
    "deductible",
    "claim_duration_days",
    "age",
    "gender",
    "marital_status",
    "occupation",
    "annual_income",
    "education_level",
    "credit_score",
    "years_with_company",
    "previous_claims",
    "policy_type",
    "policy_duration_years",
    "premium_amount",
    "coverage_amount",
    "deductible_plan",
    "payment_method",
    "policy_active",
    "incident_type",
    "accident_severity",
    "incident_location",
    "weather_condition",
    "police_report",
    "witnesses",
    "injuries",
    "hospital_visit",
    "property_damage",
    "vehicle_damage",
    "vehicle_year",
    "vehicle_age",
    "mileage",
    "vehicle_value",
    "ownership_type",
    "gps_location_match",
    "repair_shop_verified",
    "identity_verified",
    "claim_history_risk",
    "reporting_delay_days",
    "policy_age_days",
    "claim_to_coverage_ratio",
    "claim_to_income_ratio",
    "suspicious_documents_binary",
    "delayed_reporting_binary",
    "inconsistent_statements_binary",
    "claim_submitted_at_night_binary",
    "multiple_claims_last_year_binary",
    "fraud_indicator_count",
    "high_risk_combination",
]


# Values found in the supplied training dataset.
CATEGORIES = {
    "gender": ["Female", "Male", "Prefer Not to Say", "Non-Binary"],
    "marital_status": ["Single", "Married", "Widowed", "Divorced"],
    "occupation": [
        "Accountant", "Software Engineer", "Sales Representative",
        "Construction Worker", "Manager", "Other", "Doctor", "Mechanic",
        "Lawyer", "Nurse", "Retired", "Teacher", "Student", "Self-Employed"
    ],
    "education_level": ["Bachelor", "Master", "Doctorate", "High School", "Associate"],
    "policy_type": [
        "Liability Only", "Uninsured Motorist",
        "Personal Injury Protection", "Collision", "Comprehensive"
    ],
    "deductible_plan": ["Low ($250)", "High ($1,000)", "Premium ($2,500)", "Standard ($500)"],
    "payment_method": ["Auto-Debit", "Bank Transfer", "Credit Card", "Debit Card", "Check"],
    "policy_active": ["Yes", "No"],
    "incident_type": [
        "Multi-Vehicle Collision", "Single Vehicle Collision",
        "Theft", "Parked Vehicle", "Animal Impact", "Vandalism"
    ],
    "accident_severity": ["Moderate", "Total Loss", "Major", "Minor"],
    "incident_location": ["Highway", "Parking Lot", "Suburban", "Urban", "Rural"],
    "weather_condition": ["Clear", "Icy", "Rainy", "Foggy", "Snowy", "Windy"],
    "police_report": ["Yes", "No"],
    "hospital_visit": ["No", "Yes"],
    "property_damage": ["Yes", "No"],
    "vehicle_damage": [
        "Side Impact", "Rollover", "Unknown",
        "Front Bump", "Rear Bump", "Severe Structural"
    ],
    "ownership_type": ["Leased", "Owned", "Financed"],
    "gps_location_match": ["Yes", "No"],
    "repair_shop_verified": ["Yes", "No"],
    "identity_verified": ["Yes", "No"],
    "claim_history_risk": ["Medium", "Low", "High"],
}


def yes_no_binary(value):
    return 1 if value == "Yes" else 0


def build_input_dataframe(values):
    """Create the exact feature order expected by the saved models."""

    claim_amount = float(values["claim_amount"])
    annual_income = float(values["annual_income"])
    coverage_amount = float(values["coverage_amount"])

    claim_to_coverage_ratio = (
        claim_amount / coverage_amount if coverage_amount != 0 else 0.0
    )
    claim_to_income_ratio = (
        claim_amount / annual_income if annual_income != 0 else 0.0
    )

    suspicious_documents_binary = yes_no_binary(values["suspicious_documents"])
    delayed_reporting_binary = yes_no_binary(values["delayed_reporting"])
    inconsistent_statements_binary = yes_no_binary(values["inconsistent_statements"])
    claim_submitted_at_night_binary = yes_no_binary(values["claim_submitted_at_night"])
    multiple_claims_last_year_binary = yes_no_binary(values["multiple_claims_last_year"])

    fraud_indicator_count = (
        suspicious_documents_binary
        + delayed_reporting_binary
        + inconsistent_statements_binary
        + claim_submitted_at_night_binary
        + multiple_claims_last_year_binary
    )

    high_risk_combination = int(
        suspicious_documents_binary == 1
        and delayed_reporting_binary == 1
        and inconsistent_statements_binary == 1
    )

    row = {
        "claim_amount": claim_amount,
        "deductible": float(values["deductible"]),
        "claim_duration_days": int(values["claim_duration_days"]),
        "age": int(values["age"]),
        "gender": values["gender"],
        "marital_status": values["marital_status"],
        "occupation": values["occupation"],
        "annual_income": annual_income,
        "education_level": values["education_level"],
        "credit_score": int(values["credit_score"]),
        "years_with_company": int(values["years_with_company"]),
        "previous_claims": int(values["previous_claims"]),
        "policy_type": values["policy_type"],
        "policy_duration_years": int(values["policy_duration_years"]),
        "premium_amount": float(values["premium_amount"]),
        "coverage_amount": coverage_amount,
        "deductible_plan": values["deductible_plan"],
        "payment_method": values["payment_method"],
        "policy_active": values["policy_active"],
        "incident_type": values["incident_type"],
        "accident_severity": values["accident_severity"],
        "incident_location": values["incident_location"],
        "weather_condition": values["weather_condition"],
        "police_report": values["police_report"],
        "witnesses": int(values["witnesses"]),
        "injuries": int(values["injuries"]),
        "hospital_visit": values["hospital_visit"],
        "property_damage": values["property_damage"],
        "vehicle_damage": values["vehicle_damage"],
        "vehicle_year": int(values["vehicle_year"]),
        "vehicle_age": int(values["vehicle_age"]),
        "mileage": int(values["mileage"]),
        "vehicle_value": float(values["vehicle_value"]),
        "ownership_type": values["ownership_type"],
        "gps_location_match": values["gps_location_match"],
        "repair_shop_verified": values["repair_shop_verified"],
        "identity_verified": values["identity_verified"],
        "claim_history_risk": values["claim_history_risk"],
        "reporting_delay_days": int(values["reporting_delay_days"]),
        "policy_age_days": int(values["policy_age_days"]),
        "claim_to_coverage_ratio": claim_to_coverage_ratio,
        "claim_to_income_ratio": claim_to_income_ratio,
        "suspicious_documents_binary": suspicious_documents_binary,
        "delayed_reporting_binary": delayed_reporting_binary,
        "inconsistent_statements_binary": inconsistent_statements_binary,
        "claim_submitted_at_night_binary": claim_submitted_at_night_binary,
        "multiple_claims_last_year_binary": multiple_claims_last_year_binary,
        "fraud_indicator_count": fraud_indicator_count,
        "high_risk_combination": high_risk_combination,
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🛡️ Insurance Claim Fraud Detection")
st.write(
    "Enter the claim information below. The saved Random Forest model "
    "provides the primary fraud prediction, while the saved Isolation Forest "
    "model provides an independent anomaly check."
)

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
try:
    rf_model, iso_model, iso_preprocessor = load_models()
except Exception as e:
    st.error("Could not load the saved model files.")
    st.code(
        "Make sure these files are in the same folder as app.py:\n\n"
        "random_forest_fraud_model.pkl\n"
        "isolation_forest_fraud_model.pkl\n"
        "isolation_forest_preprocessor.pkl\n\n"
        "Also install the training scikit-learn version:\n"
        "scikit-learn==1.7.2"
    )
    st.exception(e)
    st.stop()


with st.form("fraud_detection_form"):

    # ========================================================
    # 1. Claim information
    # ========================================================
    st.subheader("1️⃣ Claim Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )
        deductible = st.number_input(
            "Deductible",
            min_value=0.0,
            value=500.0,
            step=50.0
        )
        claim_duration_days = st.number_input(
            "Claim Duration (days)",
            min_value=0,
            value=7,
            step=1
        )
        reporting_delay_days = st.number_input(
            "Reporting Delay (days)",
            min_value=0,
            value=2,
            step=1
        )

    with c2:
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )
        coverage_amount = st.number_input(
            "Coverage Amount",
            min_value=0.0,
            value=200000.0,
            step=5000.0
        )
        premium_amount = st.number_input(
            "Premium Amount",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )
        policy_age_days = st.number_input(
            "Policy Age (days)",
            min_value=0,
            value=1000,
            step=1
        )

    with c3:
        previous_claims = st.number_input(
            "Previous Claims",
            min_value=0,
            value=1,
            step=1
        )
        witnesses = st.number_input(
            "Witnesses",
            min_value=0,
            value=1,
            step=1
        )
        injuries = st.number_input(
            "Injuries",
            min_value=0,
            value=0,
            step=1
        )

    # ========================================================
    # 2. Customer information
    # ========================================================
    st.subheader("2️⃣ Customer Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", CATEGORIES["gender"])

    with c2:
        marital_status = st.selectbox("Marital Status", CATEGORIES["marital_status"])
        occupation = st.selectbox("Occupation", CATEGORIES["occupation"])

    with c3:
        education_level = st.selectbox("Education Level", CATEGORIES["education_level"])
        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=900,
            value=650,
            step=1
        )

    with c4:
        years_with_company = st.number_input(
            "Years With Company",
            min_value=0,
            value=5,
            step=1
        )
        claim_history_risk = st.selectbox(
            "Claim History Risk",
            CATEGORIES["claim_history_risk"]
        )

    # ========================================================
    # 3. Policy information
    # ========================================================
    st.subheader("3️⃣ Policy Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        policy_type = st.selectbox("Policy Type", CATEGORIES["policy_type"])
        policy_duration_years = st.number_input(
            "Policy Duration (years)",
            min_value=1,
            value=5,
            step=1
        )

    with c2:
        deductible_plan = st.selectbox(
            "Deductible Plan",
            CATEGORIES["deductible_plan"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            CATEGORIES["payment_method"]
        )

    with c3:
        policy_active = st.selectbox("Policy Active", CATEGORIES["policy_active"])
        police_report = st.selectbox("Police Report", CATEGORIES["police_report"])

    with c4:
        gps_location_match = st.selectbox(
            "GPS Location Match",
            CATEGORIES["gps_location_match"]
        )
        identity_verified = st.selectbox(
            "Identity Verified",
            CATEGORIES["identity_verified"]
        )

    # ========================================================
    # 4. Incident information
    # ========================================================
    st.subheader("4️⃣ Incident Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        incident_type = st.selectbox(
            "Incident Type",
            CATEGORIES["incident_type"]
        )
        accident_severity = st.selectbox(
            "Accident Severity",
            CATEGORIES["accident_severity"]
        )

    with c2:
        incident_location = st.selectbox(
            "Incident Location",
            CATEGORIES["incident_location"]
        )
        weather_condition = st.selectbox(
            "Weather Condition",
            CATEGORIES["weather_condition"]
        )

    with c3:
        hospital_visit = st.selectbox(
            "Hospital Visit",
            CATEGORIES["hospital_visit"]
        )
        property_damage = st.selectbox(
            "Property Damage",
            CATEGORIES["property_damage"]
        )

    vehicle_damage = st.selectbox(
        "Vehicle Damage",
        CATEGORIES["vehicle_damage"]
    )

    # ========================================================
    # 5. Vehicle information
    # ========================================================
    st.subheader("5️⃣ Vehicle Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        vehicle_year = st.number_input(
            "Vehicle Year",
            min_value=1950,
            max_value=2100,
            value=2020,
            step=1
        )

    with c2:
        vehicle_age = st.number_input(
            "Vehicle Age",
            min_value=0,
            value=5,
            step=1
        )

    with c3:
        mileage = st.number_input(
            "Mileage",
            min_value=0,
            value=70000,
            step=1000
        )

    with c4:
        vehicle_value = st.number_input(
            "Vehicle Value",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

    ownership_type = st.selectbox(
        "Ownership Type",
        CATEGORIES["ownership_type"]
    )

    # ========================================================
    # 6. Verification / fraud indicators
    # ========================================================
    st.subheader("6️⃣ Verification & Fraud Indicators")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        suspicious_documents = st.selectbox(
            "Suspicious Documents",
            ["No", "Yes"]
        )

    with c2:
        delayed_reporting = st.selectbox(
            "Delayed Reporting",
            ["No", "Yes"]
        )

    with c3:
        inconsistent_statements = st.selectbox(
            "Inconsistent Statements",
            ["No", "Yes"]
        )

    with c4:
        claim_submitted_at_night = st.selectbox(
            "Claim Submitted at Night",
            ["No", "Yes"]
        )

    with c5:
        multiple_claims_last_year = st.selectbox(
            "Multiple Claims Last Year",
            ["No", "Yes"]
        )

    repair_shop_verified = st.selectbox(
        "Repair Shop Verified",
        CATEGORIES["repair_shop_verified"]
    )

    st.caption(
        "The following engineered fields are calculated automatically from the "
        "values above: claim-to-coverage ratio, claim-to-income ratio, binary "
        "fraud indicators, fraud indicator count, and high-risk combination."
    )

    submitted = st.form_submit_button(
        "🔍 Predict Fraud",
        use_container_width=True
    )


# ============================================================
# Prediction
# ============================================================
if submitted:

    values = {
        "claim_amount": claim_amount,
        "deductible": deductible,
        "claim_duration_days": claim_duration_days,
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "occupation": occupation,
        "annual_income": annual_income,
        "education_level": education_level,
        "credit_score": credit_score,
        "years_with_company": years_with_company,
        "previous_claims": previous_claims,
        "policy_type": policy_type,
        "policy_duration_years": policy_duration_years,
        "premium_amount": premium_amount,
        "coverage_amount": coverage_amount,
        "deductible_plan": deductible_plan,
        "payment_method": payment_method,
        "policy_active": policy_active,
        "incident_type": incident_type,
        "accident_severity": accident_severity,
        "incident_location": incident_location,
        "weather_condition": weather_condition,
        "police_report": police_report,
        "witnesses": witnesses,
        "injuries": injuries,
        "hospital_visit": hospital_visit,
        "property_damage": property_damage,
        "vehicle_damage": vehicle_damage,
        "vehicle_year": vehicle_year,
        "vehicle_age": vehicle_age,
        "mileage": mileage,
        "vehicle_value": vehicle_value,
        "ownership_type": ownership_type,
        "gps_location_match": gps_location_match,
        "repair_shop_verified": repair_shop_verified,
        "identity_verified": identity_verified,
        "claim_history_risk": claim_history_risk,
        "reporting_delay_days": reporting_delay_days,
        "policy_age_days": policy_age_days,
        "suspicious_documents": suspicious_documents,
        "delayed_reporting": delayed_reporting,
        "inconsistent_statements": inconsistent_statements,
        "claim_submitted_at_night": claim_submitted_at_night,
        "multiple_claims_last_year": multiple_claims_last_year,
    }

    input_df = build_input_dataframe(values)

    try:
        # -----------------------------
        # Random Forest
        # -----------------------------
        rf_prediction = int(rf_model.predict(input_df)[0])
        rf_probability = None

        if hasattr(rf_model, "predict_proba"):
            probabilities = rf_model.predict_proba(input_df)[0]
            classes = list(rf_model.classes_)
            if 1 in classes:
                rf_probability = float(probabilities[classes.index(1)])

        # -----------------------------
        # Isolation Forest
        # -----------------------------
        iso_input = iso_preprocessor.transform(input_df)
        iso_prediction = int(iso_model.predict(iso_input)[0])
        iso_score = float(iso_model.decision_function(iso_input)[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            if rf_prediction == 1:
                st.error("🚨 Random Forest: FRAUD")
            else:
                st.success("✅ Random Forest: NOT FRAUD")

        with r2:
            if rf_probability is not None:
                st.metric(
                    "Random Forest Fraud Probability",
                    f"{rf_probability * 100:.2f}%"
                )
            else:
                st.metric("Random Forest Prediction", str(rf_prediction))

        with r3:
            if iso_prediction == -1:
                st.warning("⚠️ Isolation Forest: ANOMALY")
            else:
                st.success("Isolation Forest: NORMAL")

        st.subheader("🔎 Model Details")

        d1, d2 = st.columns(2)

        with d1:
            st.write("**Random Forest prediction:**", rf_prediction)
            if rf_probability is not None:
                st.write(
                    "**Fraud probability:**",
                    f"{rf_probability * 100:.2f}%"
                )

        with d2:
            st.write("**Isolation Forest prediction:**", iso_prediction)
            st.write("**Isolation Forest decision score:**", f"{iso_score:.6f}")

        st.subheader("🧮 Engineered Features Used")

        engineered = pd.DataFrame({
            "Feature": [
                "claim_to_coverage_ratio",
                "claim_to_income_ratio",
                "suspicious_documents_binary",
                "delayed_reporting_binary",
                "inconsistent_statements_binary",
                "claim_submitted_at_night_binary",
                "multiple_claims_last_year_binary",
                "fraud_indicator_count",
                "high_risk_combination",
            ],
            "Value": [
                input_df.iloc[0]["claim_to_coverage_ratio"],
                input_df.iloc[0]["claim_to_income_ratio"],
                input_df.iloc[0]["suspicious_documents_binary"],
                input_df.iloc[0]["delayed_reporting_binary"],
                input_df.iloc[0]["inconsistent_statements_binary"],
                input_df.iloc[0]["claim_submitted_at_night_binary"],
                input_df.iloc[0]["multiple_claims_last_year_binary"],
                input_df.iloc[0]["fraud_indicator_count"],
                input_df.iloc[0]["high_risk_combination"],
            ]
        })

        st.dataframe(engineered, use_container_width=True, hide_index=True)

        st.subheader("📋 Model Input")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.divider()
st.caption(
    "This application uses your saved Random Forest and Isolation Forest models. "
    "It is a machine-learning demonstration and should not be used as the sole "
    "basis for real insurance claim decisions."
)
