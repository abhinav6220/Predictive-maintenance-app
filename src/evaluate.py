import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import preprocess_data
from appliances.pump import load_data as load_pump
from appliances.hx import load_data as load_hx
from appliances.pump import get_target_column as pump_target
from appliances.hx import get_target_column as hx_target


print("Select Appliance to Evaluate")
print("1. Pump")
print("2. Heat Exchanger")

choice = input("Enter choice: ")

# ---------------- LOAD DATA ----------------

if choice == "1":
    df = load_pump("data/pump_data.csv")
    target_column = pump_target()

elif choice == "2":
    df = load_hx("data/heat_exchanger_dataset.csv")
    target_column = hx_target()

else:
    print("Invalid choice")
    exit()


# ---------------- PREPROCESS ----------------

X, y, scaler = preprocess_data(df, target_column)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ----------------

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ---------------- METRICS ----------------

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n------ MODEL PERFORMANCE ------")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)


# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ---------------- ROC CURVE ----------------

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ---------------- PRECISION RECALL CURVE ----------------

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# ---------------- PREDICTION DISTRIBUTION ----------------

plt.figure()
sns.histplot(y_prob, bins=20)
plt.title("Failure Probability Distribution")
plt.xlabel("Predicted Failure Probability")
plt.show()


# ---------------- FEATURE IMPORTANCE ----------------

importance = model.feature_importances_
features = df.drop(columns=[target_column]).columns

plt.figure()
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.show()

# ---------------- CORRELATION HEATMAP ----------------

plt.figure()

corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Sensor Correlation Heatmap")
plt.show()

# ---------------- METRICS BAR CHART ----------------

metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}

plt.figure()

sns.barplot(
    x=list(metrics.keys()),
    y=list(metrics.values())
)

plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.xlabel("Metric")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.show()