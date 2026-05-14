import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # ---------------- FEATURES ----------------

    features = [
        "Flue gas temperature (℃)",
        "Boiler oxygen level (%)",
        "CO (mg/m3)",
        "Nox (mg/m3)",
        "SO2 (mg/m3)",
        "Dust (mg/m3)",
        "Boiler Eff (%)",
        "Gross Load (MW)",
        "Coal Flow (t/h)",
    ]

    # Check missing columns
    missing = [c for c in features if c not in df.columns]

    if missing:
        raise ValueError(f"Boiler dataset missing columns: {missing}")

    # ---------------- THRESHOLD CALCULATION ----------------

    # Lower k => more detectable degradation
    k = 1.3

    thresholds = {}

    for col in features:

        series = pd.to_numeric(df[col], errors="coerce")

        mu = float(series.mean())
        sd = float(series.std(ddof=0))

        thresholds[col] = (mu, sd)

    # High abnormality
    def hi(col):
        mu, sd = thresholds[col]
        return df[col] > (mu + k * sd)

    # Low abnormality
    def lo(col):
        mu, sd = thresholds[col]
        return df[col] < (mu - k * sd)

    # ---------------- DEGRADATION SCORING ----------------

    score = (
        2 * hi("Flue gas temperature (℃)").astype(int)
        + 2 * hi("CO (mg/m3)").astype(int)
        + 2 * hi("Dust (mg/m3)").astype(int)
        + hi("Boiler oxygen level (%)").astype(int)
        + hi("Nox (mg/m3)").astype(int)
        + hi("SO2 (mg/m3)").astype(int)
        + 2 * lo("Boiler Eff (%)").astype(int)
    )

    # More realistic maintenance trigger
    maintenance = score >= 2

    # ---------------- FINAL DATAFRAME ----------------

    df = df[features].copy()

    df["Maintenance_Flag"] = maintenance.astype(int)

    # ---------------- LABEL NOISE ----------------

    noise = np.random.rand(len(df)) < 0.03

    df.loc[noise, "Maintenance_Flag"] = (
        1 - df.loc[noise, "Maintenance_Flag"]
    )

    return df


def get_target_column() -> str:
    return "Maintenance_Flag"