import pandas as pd
import numpy as np

def load_data(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Design values
    lmtd_design = df["hx_1_logarithmic_mean_temperature_difference_lmtd_k"].max()
    heat_design = df["hx_1_heat_load_kw"].max()

    # Maintenance rule
    df["Maintenance_Flag"] = (
        (df["hx_1_logarithmic_mean_temperature_difference_lmtd_k"] < 0.85 * lmtd_design) |
        (df["hx_1_heat_load_kw"] < 0.9 * heat_design)
    ).astype(int)

    # -------- ADD RANDOM NOISE --------
    noise = np.random.rand(len(df)) < 0.05
    df.loc[noise, "Maintenance_Flag"] = 1 - df.loc[noise, "Maintenance_Flag"]

    # Model features
    df = df[
        [
            "hot_inlet_temperature_k",
            "hot_outlet_temperature_k",
            "cold_outlet_temperature_k",
            "cold_inlet_mass_flow_kg_s",
            "Maintenance_Flag"
        ]
    ]

    return df


def get_target_column():
    return "Maintenance_Flag"