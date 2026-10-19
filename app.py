from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.joblib"
WEIGHTS_PATH = BASE_DIR / "best.weights.h5"
COLUMNS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
THRESHOLD = 0.5

def build_model():
    inputs = Input(shape=(20,), name="input_layer")
    x = Dense(64, activation="relu", name="dense")(inputs)
    x = Dropout(0.3, name="dropout")(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_2")(x)
    return Model(inputs=inputs, outputs=outputs, name="functional")

@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PREPROCESSOR_PATH.name}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {WEIGHTS_PATH.name}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = build_model()
    model.load_weights(str(WEIGHTS_PATH))
    return preprocessor, model

st.title("Stroke Prediction")
st.write("A web interface for the existing Stroke Prediction neural network.")
st.warning("Educational project only. This is not a medical diagnosis and must not be used for clinical decisions.")

try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age (years)", min_value=0.08, max_value=120.0, value=45.0, step=1.0)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
    with right:
        work_type = st.selectbox("Work type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
        residence_type = st.selectbox("Residence type", ["Rural", "Urban"])
        glucose = st.number_input("Average glucose level", min_value=30.0, max_value=400.0, value=95.0, step=0.5)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.1, step=0.1)
        smoking_status = st.selectbox("Smoking status", ["Unknown", "formerly smoked", "never smoked", "smokes"])
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender, "age": float(age), "hypertension": int(hypertension == "Yes"),
        "heart_disease": int(heart_disease == "Yes"), "ever_married": ever_married,
        "work_type": work_type, "Residence_type": residence_type,
        "avg_glucose_level": float(glucose), "bmi": float(bmi), "smoking_status": smoking_status
    }], columns=COLUMNS)
    try:
        transformed = preprocessor.transform(input_df)
        if transformed.shape != (1, 20):
            raise ValueError(f"Expected transformed shape (1, 20), got {transformed.shape}")
        score = float(model(transformed, training=False).numpy().ravel()[0])
        predicted_class = int(score >= THRESHOLD)
        st.subheader("Prediction result")
        st.metric("Model output score", f"{score:.4f}")
        if predicted_class == 1:
            st.error("Predicted class: 1")
        else:
            st.success("Predicted class: 0")
        st.caption("The model was trained with class weights. This score is not a calibrated clinical probability. The 0.50 threshold reproduces the notebook threshold and is not a medical decision threshold.")
        with st.expander("View input and preprocessing details"):
            st.dataframe(input_df, use_container_width=True)
            st.write("Transformed shape:", transformed.shape)
            try:
                st.dataframe(pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out()), use_container_width=True)
            except Exception:
                st.write("Feature names could not be retrieved from this preprocessor.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
