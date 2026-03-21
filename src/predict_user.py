import joblib
import numpy as np

print("Select Appliance:")
print("1. Pump")
print("2. Heat Exchanger")

choice = input("Enter choice: ")

# ===================== PUMP =====================
if choice == "1":
    model = joblib.load("models/pump/model.pkl")
    scaler = joblib.load("models/pump/scaler.pkl")

    print("\nEnter Pump Sensor Values:")

    air_temp = float(input("Air Temperature (K): "))
    process_temp = float(input("Process Temperature (K): "))
    rpm = float(input("Rotational Speed (rpm): "))
    torque = float(input("Torque (Nm): "))
    tool_wear = float(input("Tool Wear (min): "))

    # --- Pump Validation ---
    if process_temp <= air_temp:
        print("Invalid Input: Process temperature must be greater than air temperature.")
        exit()

    if rpm <= 0:
        print("Invalid Input: RPM must be greater than zero.")
        exit()

    if torque <= 0:
        print("Invalid Input: Torque must be positive.")
        exit()

    if tool_wear < 0:
        print("Invalid Input: Tool wear cannot be negative.")
        exit()

    # Optional warnings
    if air_temp < 250 or air_temp > 350:
        print("Warning: Air temperature is outside typical operating range.")

    if rpm > 4000:
        print("Warning: RPM unusually high.")

    data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])

# ===================== HX =====================
elif choice == "2":
    model = joblib.load("models/hx/model.pkl")
    scaler = joblib.load("models/hx/scaler.pkl")

    print("\nEnter Heat Exchanger Values:")

    hot_in = float(input("Hot Inlet Temperature (K): "))
    hot_out = float(input("Hot Outlet Temperature (K): "))
    cold_out = float(input("Cold Outlet Temperature (K): "))
    cold_flow = float(input("Cold Inlet Mass Flow (kg/s): "))
    # heat_load = float(input("Heat Load (kW): "))
    # lmtd = float(input("LMTD (K): "))

    # --- HX Validation ---
    if hot_in <= hot_out:
        print("Invalid Input: Hot inlet temperature must be greater than hot outlet temperature.")
        exit()

    if hot_out <= cold_out:
        print("Invalid Input: Hot outlet temperature must be greater than cold outlet temperature.")
        exit()

    if hot_in <= 0 or hot_out <= 0 or cold_out <= 0:
        print("Invalid Input: Temperatures must be positive.")
        exit()

    if cold_flow <= 0:
        print("Invalid Input: Mass flow rate must be positive.")
        exit()

    # if heat_load <= 0:
        print("Invalid Input: Heat load must be positive.")
        exit()

    # if lmtd <= 0:
        print("Invalid Input: LMTD must be positive.")
        exit()

    data = np.array([[hot_in, hot_out, cold_out, cold_flow]])

# ===================== INVALID =====================
else:
    print("Invalid choice.")
    exit()

# ===================== PREDICTION =====================

data_scaled = scaler.transform(data)

prediction = model.predict(data_scaled)
probability = model.predict_proba(data_scaled)

print("\n--- RESULT ---")

if prediction[0] == 1:
    print("⚠ Maintenance Required")
else:
    print("✅ System Healthy")

print("Failure Probability:", round(probability[0][1] * 100, 2), "%")