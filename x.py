import pandas as pd

df = pd.read_csv("data/gas_turbine_dataset.csv")

print(df["Fault"].value_counts())