# Predictive Maintenance Web App

Streamlit app for Pump and Heat Exchanger failure prediction with risk badge, risk flags, and report download (CSV/PDF).

## Run locally

```powershell
cd "d:\web deve\predictive_maintenance"
d:\web deve\.venv\Scripts\python.exe -m streamlit run src/app_ui.py
```

App URL:

- `http://localhost:8501`

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Confirm these paths exist in the repo:
   - `src/app_ui.py`
   - `requirements.txt` (includes pinned `altair<5` for Streamlit compatibility)
   - `runtime.txt` (uses Python 3.12 on Streamlit Cloud for stability)
   - `models/pump/model.pkl`
   - `models/pump/scaler.pkl`
   - `models/hx/model.pkl`
   - `models/hx/scaler.pkl`
3. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **New app** and choose:
   - **Repository**: your repo
   - **Branch**: main (or your branch)
   - **Main file path**: `src/app_ui.py`
5. Click **Deploy**.

After build, Streamlit gives a public app URL.

## Notes

- This app does not need a database for prediction.
- Database is optional and only needed for features like user accounts or permanent prediction history.
- If PDF download fails in a local environment, install dependency in the same runtime:

```powershell
python -m pip install fpdf2
```
