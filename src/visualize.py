import matplotlib.pyplot as plt
import seaborn as sns

from appliances.pump import load_data as load_pump, get_target_column as pump_target
from appliances.hx import load_data as load_hx, get_target_column as hx_target


def visualize_pump():
    df = load_pump("data/pump_data.csv")
    target_col = pump_target()

    feature_cols = [c for c in df.columns if c != target_col]

    print("Pump data columns:", df.columns.tolist())
    print("Using target column:", target_col)

    sns.set(style="whitegrid")
    df[feature_cols].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Pump – Feature Distributions")
    plt.tight_layout()
    plt.show()

    counts = df[target_col].value_counts().sort_index()
    labels = ["Healthy (0)", "Failure (1)"]

    plt.figure(figsize=(5, 4))
    sns.barplot(x=labels, y=counts.values)
    plt.title("Pump – Failure vs Healthy")
    plt.ylabel("Count")
    plt.show()


def visualize_hx():
    df = load_hx("data/heat_exchanger_dataset.csv")
    target_col = hx_target()

    feature_cols = [c for c in df.columns if c != target_col]

    print("HX data columns:", df.columns.tolist())
    print("Using target column:", target_col)

    sns.set(style="whitegrid")
    df[feature_cols].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Heat Exchanger – Feature Distributions")
    plt.tight_layout()
    plt.show()

    counts = df[target_col].value_counts().sort_index()
    labels = ["Healthy (0)", "Maintenance Needed (1)"]

    plt.figure(figsize=(5, 4))
    sns.barplot(x=labels, y=counts.values)
    plt.title("Heat Exchanger – Maintenance Flag Distribution")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    print("Select Appliance to Visualize:")
    print("1. Pump")
    print("2. Heat Exchanger")

    choice = input("Enter choice: ")

    if choice == "1":
        visualize_pump()
    elif choice == "2":
        visualize_hx()
    else:
        print("Invalid choice.")

