
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np

# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Smart Doctor API",
    description="Fog Computing Patient Monitoring System",
    version="1.0"
)

# Allow the Netlify frontend to access this backend.
# We can restrict this to your exact Netlify domain later.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. FOG PROCESSING FUNCTION
# ==========================================

def process_patient_data(row):

    risk_score = 0
    critical_condition = False
    reasons = []

    # Heart rate
    if row["Heart_Rate"] >= 130 or row["Heart_Rate"] < 40:
        risk_score += 3
        critical_condition = True
        reasons.append("Critical heart rate")

    elif row["Heart_Rate"] > 100 or row["Heart_Rate"] < 50:
        risk_score += 1
        reasons.append("Abnormal heart rate")

    # Oxygen saturation
    if row["SpO2"] <= 90:
        risk_score += 3
        critical_condition = True
        reasons.append("Critical oxygen saturation")

    elif row["SpO2"] <= 94:
        risk_score += 2
        reasons.append("Low oxygen saturation")

    # Temperature
    if row["Temperature"] >= 39:
        risk_score += 3
        critical_condition = True
        reasons.append("High temperature")

    elif row["Temperature"] >= 38:
        risk_score += 1
        reasons.append("Elevated temperature")

    # Systolic blood pressure
    if row["Systolic_BP"] >= 180:
        risk_score += 3
        critical_condition = True
        reasons.append("Critical systolic blood pressure")

    elif row["Systolic_BP"] >= 140:
        risk_score += 2
        reasons.append("Elevated systolic blood pressure")

    # Risk classification
    if critical_condition or risk_score >= 5:
        status = "CRITICAL"

    elif risk_score >= 2:
        status = "WARNING"

    else:
        status = "NORMAL"

    alert = (
        ", ".join(reasons)
        if reasons
        else "No simulated abnormalities detected"
    )

    return pd.Series([risk_score, status, alert])


# ==========================================
# 3. GENERATE SIMULATED PATIENT DATA
# ==========================================

def generate_patient_data():

    heart_rates = np.random.randint(65, 100, 10)
    spo2_values = np.random.randint(95, 100, 10)
    temperatures = np.round(
        np.random.uniform(36.2, 37.5, 10), 1
    )
    blood_pressure = np.random.randint(110, 135, 10)

    # Simulated abnormal readings for P003

    heart_rates[2] = np.random.randint(135, 150)
    spo2_values[2] = np.random.randint(85, 90)
    temperatures[2] = 39.5
    blood_pressure[2] = 175

    # Simulated abnormal readings for P005

    heart_rates[4] = np.random.randint(105, 115)
    spo2_values[4] = 93
    temperatures[4] = 38.2
    blood_pressure[4] = 145

    patient_data = pd.DataFrame({
        "Patient_ID": [
            f"P{i:03d}" for i in range(1, 11)
        ],
        "Heart_Rate": heart_rates,
        "SpO2": spo2_values,
        "Temperature": temperatures,
        "Systolic_BP": blood_pressure,
        "Timestamp": datetime.now().isoformat()
    })

    # Fog layer processes patient data locally

    processed_data = patient_data.copy()

    processed_data[
        ["Risk_Score", "Status", "Alert"]
    ] = patient_data.apply(
        process_patient_data,
        axis=1
    )

    # Alert prioritization

    priority_order = {
        "CRITICAL": 1,
        "WARNING": 2,
        "NORMAL": 3
    }

    processed_data["Priority"] = (
        processed_data["Status"].map(priority_order)
    )

    priority_data = processed_data.sort_values(
        by=["Priority", "Risk_Score"],
        ascending=[True, False]
    )

    return processed_data, priority_data


# ==========================================
# 4. HOME API
# ==========================================

@app.get("/")
def home():

    return {
        "project": "Smart Doctor",
        "message": "Fog Computing Backend is Running",
        "status": "Success"
    }


# ==========================================
# 5. LIVE PATIENT DATA API
# ==========================================

@app.get("/live-data")
def get_live_data():

    processed_data, priority_data = (
        generate_patient_data()
    )

    patients = priority_data.drop(
        columns=["Priority", "Timestamp"]
    ).to_dict(orient="records")

    return {
        "patients": patients,

        "alerts": {
            "critical": int(
                (processed_data["Status"] == "CRITICAL").sum()
            ),

            "warning": int(
                (processed_data["Status"] == "WARNING").sum()
            ),

            "normal": int(
                (processed_data["Status"] == "NORMAL").sum()
            )
        },

        "last_updated": datetime.now().isoformat()
    }


# ==========================================
# 6. PATIENT API
# ==========================================

@app.get("/patients")
def get_patients():

    processed_data, priority_data = (
        generate_patient_data()
    )

    return priority_data[
        [
            "Patient_ID",
            "Heart_Rate",
            "SpO2",
            "Temperature",
            "Systolic_BP",
            "Risk_Score",
            "Status",
            "Alert"
        ]
    ].to_dict(orient="records")


# ==========================================
# 7. ALERT SUMMARY API
# ==========================================

@app.get("/alerts")
def get_alerts():

    processed_data, priority_data = (
        generate_patient_data()
    )

    return {
        "critical": int(
            (processed_data["Status"] == "CRITICAL").sum()
        ),

        "warning": int(
            (processed_data["Status"] == "WARNING").sum()
        ),

        "normal": int(
            (processed_data["Status"] == "NORMAL").sum()
        )
    }
