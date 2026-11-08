
import streamlit as st
import joblib
import gdown
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path("/tmp/machinery_predictive_maintenance_model.pkl")

FEATURES_PATH = BASE_DIR / "machinery_model_features.pkl"

# Your Google Drive model ID
FILE_ID = "1zrg_NMzi3UNucRIX3T8G-bUvjkBwZHI3"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        with st.spinner("Downloading trained machinery model..."):

            downloaded = gdown.download(
                id=FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
                fuzzy=True
            )

            if downloaded is None:
                raise RuntimeError("Google Drive model download failed.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file was not downloaded.")

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model appears incomplete.")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


try:
    model, features = load_model()
    st.success("Machinery predictive maintenance model loaded!")
    
# ==========================================
# MACHINERY PREDICTIVE MAINTENANCE DASHBOARD
# ==========================================

st.title("⚙️ Machinery Predictive Maintenance")
st.subheader("AI-Based Equipment Health Monitoring")

st.write(
    "Monitor machinery operating conditions and "
    "assess equipment failure classification using "
    "your trained Random Forest model."
)

st.divider()

# Equipment selection
equipment = st.selectbox(
    "Select Equipment",
    ["Centrifugal Pump", "Gas Turbine"]
)

fault_modes = [
    "Cavitation",
    "Combustion Instability",
    "Compressor Degradation",
    "Compressor Fouling",
    "Impeller Degradation",
    "Lubrication Degradation",
    "Lubrication Problem",
    "Mechanical Seal Failure",
    "Normal",
    "Rotor Imbalance",
    "Seal Degradation",
    "Shaft Misalignment",
    "Turbine Degradation"
]

fault = st.selectbox(
    "Observed Fault Mode",
    fault_modes
)

st.header("Operating Parameters")

defaults = {
    "operating_hours": 10000,
    "load_pct": 75,
    "rpm": 1800,
    "suction_pressure_bar": 2,
    "discharge_pressure_bar": 8,
    "suction_temp_C": 30,
    "discharge_temp_C": 85,
    "bearing_temp_C": 65,
    "lube_oil_temp_C": 55,
    "lube_oil_pressure_bar": 3,
    "vibration_mm_s": 2.5,
    "axial_vibration_mm_s": 1,
    "radial_vibration_mm_s": 1.5,
    "motor_current_A": 45,
    "motor_power_kW": 30,
    "flow_rate_m3_h": 50,
    "efficiency_pct": 85,
    "ambient_temp_C": 30,
    "cooling_water_temp_C": 25,
    "cooling_water_flow_m3_h": 20,
    "oil_debris_count": 5,
    "seal_leakage_rate_L_h": 0.1,
    "days_since_maintenance": 90,
    "maintenance_count": 3,
    "degradation_index": 0.2,
    "health_index": 0.8,
    "failure_next_7d": 0,
    "rul_hours": 0
}

values = {}

left, right = st.columns(2)

for i, (name, default) in enumerate(defaults.items()):
    with (left if i % 2 == 0 else right):
        values[name] = st.number_input(
            name.replace("_", " ").title(),
            value=float(default),
            key=name
        )

suction = values["suction_pressure_bar"]

values["pressure_ratio"] = (
    values["discharge_pressure_bar"] / suction
    if suction > 0 else 0
)

# Categorical feature encoding
for name in features:

    if name.startswith("equipment_type_"):
        values[name] = float(
            name == f"equipment_type_{equipment}"
        )

    elif name.startswith("fault_mode_"):
        category = name.replace("fault_mode_", "", 1)
        values[name] = float(category == fault)

# Construct model input
missing = [f for f in features if f not in values]

if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

input_data = pd.DataFrame(
    [[values[f] for f in features]],
    columns=features
)

st.divider()

# Prediction
if st.button(
    "RUN PREDICTIVE MAINTENANCE ASSESSMENT",
    type="primary",
    use_container_width=True
):

    try:
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                str(prediction)
            )

        with col2:
            st.metric(
                "Highest Probability",
                f"{max(probabilities)*100:.2f}%"
            )

        st.subheader("Prediction Probabilities")

        results = pd.DataFrame({
            "Class": model.classes_,
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            results.set_index("Class")
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.info(
            "Confirm the meaning of the model classes "
            "using the original training labels before "
            "interpreting the result as a failure warning."
        )

        st.subheader("Equipment Input Data")

        st.dataframe(
            input_data.T,
            use_container_width=True
        )

        st.download_button(
            "Download Assessment",
            input_data.to_csv(index=False),
            file_name="machinery_assessment.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

st.caption(
    "Machinery Predictive Maintenance | "
    "Random Forest Classifier | Engineering Prototype"
)


except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
