import numpy as np
import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    # Clean spaces
    df.columns = df.columns.str.strip()

    # ---------------- FEATURES ----------------

    features = [
        "Temperature (°C)",
        "RPM",
        "Torque (Nm)",
        "Vibrations (mm/s)",
        "Power Output (MW)",
        "Fuel Flow Rate (kg/s)",
        "Air Pressure (kPa)",
        "Exhaust Gas Temperature (°C)",
        "Oil Temperature (°C)",
        "Fault"
    ]

    # Check missing columns
    missing = [c for c in features if c not in df.columns]

    if missing:
        raise ValueError(f"Gas turbine dataset missing columns: {missing}")

    df = df[features].copy()

    # ---------------- FEATURE ENGINEERING ----------------

    df["Thermal_Stress"] = (
        df["Temperature (°C)"]
        * df["Exhaust Gas Temperature (°C)"]
    )

    df["Mechanical_Stress"] = (
        df["RPM"]
        * df["Vibrations (mm/s)"]
    )

    df["Combustion_Index"] = (
        df["Fuel Flow Rate (kg/s)"]
        / (df["Air Pressure (kPa)"] + 1)
    )

    df["Oil_Risk"] = (
        df["Oil Temperature (°C)"]
        * df["Vibrations (mm/s)"]
    )

    # ---------------- PHYSICS RISK ENHANCEMENT ----------------

    risk_score = (
        (df[features[0]] > 1050).astype(int)
        + (df["Vibrations (mm/s)"] > 3).astype(int)
        + (df["Oil Temperature (°C)"] > 140).astype(int)
        + (df["Exhaust Gas Temperature (°C)"] > 560).astype(int)
        + (df["Air Pressure (kPa)"] < 105).astype(int)
    )

    # Strengthen fault labels using physics
    df.loc[risk_score >= 2, "Fault"] = 1

    critical_risk = (
        (df[features[0]] > 1200)
        | (df["RPM"] > 25000)
        | (df["Vibrations (mm/s)"] > 5)
        | (df[features[8]] > 150)
        | (df[features[7]] > 700)
        | (df["Air Pressure (kPa)"] < 80)
    )

    df.loc[critical_risk, "Fault"] = 1

    # ---------------- ADD SMALL SENSOR NOISE ----------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        if col != "Fault":

            noise = np.random.normal(
                0,
                0.01 * df[col].std(),
                size=len(df)
            )

            df[col] += noise

    return df


def get_target_column():
    return "Fault"
