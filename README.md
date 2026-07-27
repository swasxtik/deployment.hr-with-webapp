# HR Salary Prediction App

A Streamlit web app that predicts an employee's salary from their age, education level, and months of experience, using a pre-trained regression model.

🔗 **Live app:** [https://deploymenthr-with-webapp-52jmzykzrfx2jhpgrgdp6e.streamlit.app/](https://deploymenthr-with-webapp-52jmzykzrfx2jhpgrgdp6e.streamlit.app/)

## Overview

This repo trains a salary-prediction model on HR data and serves it through a simple Streamlit form: enter age, education level, and months of experience, and the app returns a predicted salary.

## Repo structure

```
.
├── hrapp.py                                 # Streamlit web app (entry point)
├── hr.pkl                                   # Trained model (pickled)
├── schr.pkl                                 # Fitted feature scaler (pickled)
├── hrdataset.csv                            # Training dataset
├── hr_analytics code with window app.ipynb  # Notebook: EDA, model training, and a Tkinter desktop version of the app
├── requirements.txt                         # Python dependencies
└── .devcontainer/                           # Dev container config
```

## How it works

1. `hrapp.py` loads `hr.pkl` (model) and `schr.pkl` (scaler) at startup.
2. The user enters three inputs in the browser:
   - **Age**
   - **Education Level** — `High School or less`, `Intermediate`, `Graduation`, `PG`
   - **Months of Experience**
3. Education is mapped to a numeric code (`0`–`3`), the three features are assembled into a vector, scaled with `schr.pkl`, and passed to the model.
4. The predicted salary is displayed on the page.

## Requirements

- Python 3.9+
- See `requirements.txt`:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `joblib`
  - `matplotlib`
  - `seaborn`

> **Note:** `hr.pkl` was trained as an XGBoost model, so `xgboost` must be installed — without it, unpickling `hr.pkl` will fail with `ModuleNotFoundError`.

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/swasxtik/deployment.hr-with-webapp.git
cd deployment.hr-with-webapp

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run hrapp.py
```

The app will open at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done: `swasxtik/deployment.hr-with-webapp`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Set:
   - **Repository:** `swasxtik/deployment.hr-with-webapp`
   - **Branch:** `main`
   - **Main file path:** `hrapp.py`
5. Click **Deploy** and check the build logs if anything fails.

## Model training

The training workflow (data exploration, preprocessing, model fitting, and an earlier Tkinter-based desktop version of the app) lives in `hr_analytics code with window app.ipynb`.

## Notes / limitations

- Inputs are not validated beyond simple min/max bounds — extreme or unusual combinations may produce unreliable predictions.
- `requirements.txt` does not pin exact versions; if predictions look off after an environment rebuild, check that installed `scikit-learn`/`xgboost` versions match the versions used to train `hr.pkl`.
