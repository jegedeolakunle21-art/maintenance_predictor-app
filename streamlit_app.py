
import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "machinery_predictive_maintenance_model.pkl"
FEATURES_PATH = BASE_DIR / "machinery_model_features.pkl"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features

try:
    model, features = load_model()
    st.success("Machine learning model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
