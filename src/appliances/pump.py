import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df = df[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Machine failure"
        ]
    ]

    return df

def get_target_column():
    return "Machine failure"