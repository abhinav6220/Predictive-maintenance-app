from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Select Appliance:")
print("1. Pump")
print("2. Heat Exchanger")

choice = input("Enter choice: ")

if choice == "1":
    from appliances.pump import load_data, get_target_column
    data_path = "data/pump_data.csv"
    save_path = "models/pump/"
elif choice == "2":
    from appliances.hx import load_data, get_target_column
    data_path = "data/heat_exchanger_dataset.csv"
    save_path = "models/hx/"

df = load_data(data_path)
target_column = get_target_column()

X, y, scaler = preprocess_data(df, target_column)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

os.makedirs(save_path, exist_ok=True)

joblib.dump(model, save_path + "model.pkl")
joblib.dump(scaler, save_path + "scaler.pkl")

print("Model Trained Successfully.")