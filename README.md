# Stroke Prediction Deployment

This Streamlit app uses your existing Keras weights and fitted scikit-learn preprocessor.

## Add your existing files
Copy `best.weights.h5` and `preprocessor.joblib` into this folder. They are not included in this starter bundle.

## Windows setup (PowerShell)
Use 64-bit Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_deployment.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

The verification script uses example records to test loading and shapes. They are not guaranteed to match exact rows from the original notebook. For numerical equivalence, compare exact input rows and scores from the notebook.

Educational project only. The model score is not a calibrated clinical probability or a medical diagnosis.
