
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

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
