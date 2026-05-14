import joblib
import pandas as pd

print("Select Appliance:")
print("1. Pump")
print("2. Heat Exchanger")
print("3. Boiler")
print("4.Gas_turbine")
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

    data = pd.DataFrame(
        [[air_temp, process_temp, rpm, torque, tool_wear]],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
    )

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

  
    data = pd.DataFrame(
        [[hot_in, hot_out, cold_out, cold_flow]],
        columns=[
            "hot_inlet_temperature_k",
            "hot_outlet_temperature_k",
            "cold_outlet_temperature_k",
            "cold_inlet_mass_flow_kg_s",
        ],
    )

# ===================== BOILER =====================
elif choice == "3":
    model = joblib.load("models/boiler/model.pkl")
    scaler = joblib.load("models/boiler/scaler.pkl")

    print("\nEnter Boiler Values:")

    flue_temp = float(input("Flue gas temperature (°C): "))
    boiler_o2 = float(input("Boiler oxygen level (%): "))
    co = float(input("CO (mg/m3): "))
    nox = float(input("Nox (mg/m3): "))
    so2 = float(input("SO2 (mg/m3): "))
    dust = float(input("Dust (mg/m3): "))
    eff = float(input("Boiler Efficiency (%): "))
    gross_load = float(input("Gross Load (MW): "))
    coal_flow = float(input("Coal Flow (t/h): "))

    # --- Boiler Validation (basic sanity) ---
    if flue_temp <= 0:
        print("Invalid Input: Flue gas temperature must be positive.")
        exit()
    if boiler_o2 < 0 or boiler_o2 > 25:
        print("Invalid Input: Boiler oxygen level must be between 0 and 25.")
        exit()
    if eff <= 0 or eff > 100:
        print("Invalid Input: Boiler efficiency must be between 0 and 100.")
        exit()
    if gross_load <= 0:
        print("Invalid Input: Gross load must be positive.")
        exit()
    if coal_flow <= 0:
        print("Invalid Input: Coal flow must be positive.")
        exit()

    data = pd.DataFrame(
        [[flue_temp, boiler_o2, co, nox, so2, dust, eff, gross_load, coal_flow]],
        columns=[
            "Flue gas temperature (℃)",
            "Boiler oxygen level (%)",
            "CO (mg/m3)",
            "Nox (mg/m3)",
            "SO2 (mg/m3)",
            "Dust (mg/m3)",
            "Boiler Eff (%)",
            "Gross Load (MW)",
            "Coal Flow (t/h)",
        ],
    )


# ===================== GAS TURBINE =====================

elif choice == "4":

    model = joblib.load("models/gas_turbine/model.pkl")
    scaler = joblib.load("models/gas_turbine/scaler.pkl")

    print("\nEnter Gas Turbine Sensor Values:")
    gas_turbine_risk_reasons = []
    gas_turbine_critical_reasons = []

    temperature = float(input("Temperature (°C): "))
    rpm = float(input("RPM: "))
    torque = float(input("Torque (Nm): "))
    vibration = float(input("Vibrations (mm/s): "))
    power = float(input("Power Output (MW): "))
    fuel = float(input("Fuel Flow Rate (kg/s): "))
    air_pressure = float(input("Air Pressure (kPa): "))
    exhaust = float(input("Exhaust Gas Temperature (°C): "))
    oil_temp = float(input("Oil Temperature (°C): "))
    
               
    # ===================== GAS TURBINE VALIDATION =====================

    if temperature < 500:
        print("Warning: Turbine temperature outside normal operating range.")

    elif temperature > 1200:
        print("Warning: Turbine temperature critical.")
        gas_turbine_risk_reasons.append("turbine temperature is high")
        gas_turbine_critical_reasons.append("turbine temperature is critically high")

    elif temperature > 1050:
        print("Warning: Turbine temperature high.")
        gas_turbine_risk_reasons.append("turbine temperature is high")

    if rpm <= 0:
        print("Invalid Input: RPM must be greater than zero.")
        exit()

    if rpm > 25000:
        print("Warning: RPM unusually high.")
        gas_turbine_critical_reasons.append("rotational speed is unusually high")

    if torque <= 0:
        print("Invalid Input: Torque must be positive.")
        exit()

    if vibration < 0:
        print("Invalid Input: Vibration cannot be negative.")
        exit()

    if vibration > 3:
        print("Warning: Excessive vibration detected.")
        gas_turbine_risk_reasons.append("vibration is excessive")

    if vibration > 5:
        gas_turbine_critical_reasons.append("vibration is critically high")

    if power <= 0:
        print("Invalid Input: Power output must be positive.")
        exit()

    if fuel <= 0:
        print("Invalid Input: Fuel flow must be positive.")
        exit()

    if air_pressure <= 0:
        print("Invalid Input: Air pressure must be positive.")
        exit()

    if air_pressure < 105:
        print("Warning: Compressor air pressure unusually low.")
        gas_turbine_risk_reasons.append("compressor air pressure is too low")

    if air_pressure < 80:
        gas_turbine_critical_reasons.append("compressor air pressure is critically low")

    if exhaust < 300:
        print("Warning: Exhaust temperature unusually low.")

    if exhaust > 560:
        print("Warning: Exhaust temperature critical.")
        gas_turbine_risk_reasons.append("exhaust gas temperature is high")

    if exhaust > 700:
        gas_turbine_critical_reasons.append("exhaust gas temperature is critically high")

    if oil_temp <= 0:
        print("Invalid Input: Oil temperature must be positive.")
        exit()

    if oil_temp > 140:
        print("Warning: Oil temperature critically high.")
        gas_turbine_risk_reasons.append("oil temperature is high")

    if oil_temp > 150:
        gas_turbine_critical_reasons.append("oil temperature is critically high")

    # ===================== DATAFRAME =====================

    # ===================== FEATURE ENGINEERING =====================

    thermal_stress = (
        temperature * exhaust
    )

    mechanical_stress = (
        rpm * vibration
    )

    combustion_index = (
        fuel / (air_pressure + 1)
    )

    oil_risk = (
        oil_temp * vibration
    )

    # ===================== DATAFRAME =====================

    data = pd.DataFrame(
        [[
            temperature,
            rpm,
            torque,
            vibration,
            power,
            fuel,
            air_pressure,
            exhaust,
            oil_temp,
            thermal_stress,
            mechanical_stress,
            combustion_index,
            oil_risk
        ]],
        columns=[
            "Temperature (°C)",
            "RPM",
            "Torque (Nm)",
            "Vibrations (mm/s)",
            "Power Output (MW)",
            "Fuel Flow Rate (kg/s)",
            "Air Pressure (kPa)",
            "Exhaust Gas Temperature (°C)",
            "Oil Temperature (°C)",
            "Thermal_Stress",
            "Mechanical_Stress",
            "Combustion_Index",
            "Oil_Risk"
        ]
    )
    # ===================== INVALID CHOICE =====================

else:
    print("Invalid choice.")
    exit()









# ===================== PREDICTION =====================

# ===================== PREDICTION =====================

data_scaled = scaler.transform(data)

prediction = model.predict(data_scaled)

probability = model.predict_proba(data_scaled)

failure_prob = probability[0][1]

print("\n--- RESULT ---")

# Custom industrial threshold
maintenance_required = failure_prob >= 0.30

if choice == "4":
    physics_required = len(gas_turbine_risk_reasons) >= 2
    critical_required = len(gas_turbine_critical_reasons) >= 1
    maintenance_required = maintenance_required or physics_required or critical_required
    physics_reason_needed = failure_prob < 0.30 and (physics_required or critical_required)
else:
    physics_reason_needed = False

if maintenance_required:
    print("Maintenance Required")
else:
    print("System Healthy")

print("Failure Probability:", round(failure_prob * 100, 2), "%")

if physics_reason_needed:
    reason_parts = gas_turbine_critical_reasons or gas_turbine_risk_reasons
    print("Reason: Maintenance is required because", ", ".join(reason_parts) + ".")
