from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Select Appliance:")
print("1. Pump")
print("2. Heat Exchanger")
print("3. Boiler")
print("4. Gas_turbine")

choice = input("Enter choice: ")

# ===================== APPLIANCE SELECTION =====================

if choice == "1":

    from appliances.pump import load_data, get_target_column

    data_path = os.path.join(BASE_DIR, "data", "pump_data.csv")
    save_path = os.path.join(BASE_DIR, "models", "pump")

elif choice == "2":

    from appliances.hx import load_data, get_target_column

    data_path = os.path.join(BASE_DIR, "data", "heat_exchanger_dataset.csv")
    save_path = os.path.join(BASE_DIR, "models", "hx")

elif choice == "3":

    from appliances.boiler import load_data, get_target_column

    data_path = os.path.join(BASE_DIR, "data", "Boiler Dataset.csv")
    save_path = os.path.join(BASE_DIR, "models", "boiler")

elif choice == "4":

    from appliances.gas_turbine import load_data, get_target_column

    data_path = os.path.join(BASE_DIR, "data", "gas_turbine_dataset.csv")
    save_path = os.path.join(BASE_DIR, "models", "gas_turbine")

else:
    print("Invalid choice.")
    exit()

# ===================== LOAD DATA =====================

df = load_data(data_path)

target_column = get_target_column()

X, y, scaler = preprocess_data(df, target_column)

# ===================== TRAIN TEST SPLIT =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================== MODEL SELECTION =====================

model = XGBClassifier(
    eval_metric='logloss'
)

# ===================== TRAIN MODEL =====================

model.fit(X_train, y_train)

# ===================== SAVE MODEL =====================

os.makedirs(save_path, exist_ok=True)

joblib.dump(model, os.path.join(save_path, "model.pkl"))
joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

# ===================== TEST ACCURACY =====================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Trained Successfully.")
print("Accuracy:", round(accuracy * 100, 2), "%")
