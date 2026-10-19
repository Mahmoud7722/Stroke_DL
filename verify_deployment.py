from pathlib import Path
import joblib
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.joblib"
WEIGHTS_PATH = BASE_DIR / "best.weights.h5"
COLUMNS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]

def build_model():
    inputs = Input(shape=(20,), name="input_layer")
    x = Dense(64, activation="relu", name="dense")(inputs)
    x = Dropout(0.3, name="dropout")(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_2")(x)
    return Model(inputs=inputs, outputs=outputs, name="functional")

def main():
    print("1. Checking required files...")
    assert PREPROCESSOR_PATH.exists(), f"Missing {PREPROCESSOR_PATH.name}"
    assert WEIGHTS_PATH.exists(), f"Missing {WEIGHTS_PATH.name}"
    print("[PASS] Both artifact files exist.")

    print("\\n2. Loading preprocessor and model weights...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = build_model()
    model.load_weights(str(WEIGHTS_PATH))
    print("[PASS] Preprocessor and weights loaded.")
    print("Preprocessor input features:", getattr(preprocessor, "n_features_in_", "unknown"))
    print("Model parameters:", model.count_params())

    print("\\n3. Testing two example records...")
    samples = pd.DataFrame([
        {"gender":"Male","age":67.0,"hypertension":0,"heart_disease":1,"ever_married":"Yes","work_type":"Private","Residence_type":"Urban","avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"},
        {"gender":"Female","age":35.0,"hypertension":0,"heart_disease":0,"ever_married":"Yes","work_type":"Self-employed","Residence_type":"Rural","avg_glucose_level":82.99,"bmi":30.6,"smoking_status":"never smoked"},
    ], columns=COLUMNS)
    transformed = preprocessor.transform(samples)
    assert transformed.shape == (2, 20), f"Expected (2, 20), got {transformed.shape}"
    print("[PASS] Preprocessor output shape:", transformed.shape)
    predictions = model(transformed, training=False).numpy()
    assert predictions.shape == (2, 1), f"Expected (2, 1), got {predictions.shape}"
    print("[PASS] Model output shape:", predictions.shape)
    for i, score in enumerate(predictions.ravel(), start=1):
        print(f"Sample {i}: score={score:.6f}, class={int(score >= 0.5)}")

    print("\\n4. Testing unseen category handling...")
    unseen = samples.iloc[[0]].copy()
    unseen["gender"] = "Other"
    unseen_transformed = preprocessor.transform(unseen)
    assert unseen_transformed.shape == (1, 20), f"Unexpected shape: {unseen_transformed.shape}"
    print("[PASS] Unseen category did not break preprocessing.")
    print("\\nAll local checks passed.")
    print("Note: example records are smoke tests, not verified reference rows from the notebook.")

if __name__ == "__main__":
    main()
