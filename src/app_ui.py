from pathlib import Path
from datetime import datetime
import sys

import altair as alt
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_pump_assets():
    model = joblib.load(MODEL_DIR / "pump" / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "pump" / "scaler.pkl")
    return model, scaler


@st.cache_resource
def load_hx_assets():
    model = joblib.load(MODEL_DIR / "hx" / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "hx" / "scaler.pkl")
    return model, scaler


@st.cache_resource
def load_boiler_assets():
    model = joblib.load(MODEL_DIR / "boiler" / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "boiler" / "scaler.pkl")
    return model, scaler


@st.cache_resource
def load_gas_turbine_assets():
    model = joblib.load(MODEL_DIR / "gas_turbine" / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "gas_turbine" / "scaler.pkl")
    return model, scaler


PUMP_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

HX_FEATURES = [
    "hot_inlet_temperature_k",
    "hot_outlet_temperature_k",
    "cold_outlet_temperature_k",
    "cold_inlet_mass_flow_kg_s",
]

BOILER_FEATURES = [
    "Flue gas temperature (℃)",
    "Boiler oxygen level (%)",
    "CO (mg/m3)",
    "Nox (mg/m3)",
    "SO2 (mg/m3)",
    "Dust (mg/m3)",
    "Boiler Eff (%)",
    "Gross Load (MW)",
    "Coal Flow (t/h)",
]

GAS_TURBINE_FEATURES = [
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
    "Oil_Risk",
]


def style_app():
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="🛠️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #1e293b 0%, #0b1022 48%, #080d1c 100%);
                color: #f8fafc;
            }
            .block-container {
                max-width: 980px;
                padding-top: 1.6rem;
                padding-bottom: 1.4rem;
            }
            header[data-testid="stHeader"] {
                background: transparent;
            }
            header[data-testid="stHeader"] [data-testid="stStatusWidget"],
            header[data-testid="stHeader"] .stAppDeployButton {
                display: none !important;
                visibility: hidden !important;
            }
            div[data-testid="stToolbar"] {
                right: 1rem;
            }
            [data-testid="stForm"] {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 16px;
                padding: 1rem 1rem 0.6rem 1rem;
                backdrop-filter: blur(4px);
            }
            label, .stMarkdown p, [data-testid="stWidgetLabel"] p {
                color: #f1f5f9 !important;
                font-weight: 600 !important;
                opacity: 1 !important;
            }
            [data-testid="stCaptionContainer"] {
                color: #e2e8f0 !important;
            }
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 12px;
                padding: 0.7rem 0.9rem;
            }
            [data-testid="stMetric"] label,
            [data-testid="stMetricLabel"] {
                color: #e2e8f0 !important;
                font-weight: 700 !important;
                opacity: 1 !important;
            }
            [data-testid="stMetricValue"] {
                color: #ffffff !important;
                font-weight: 800 !important;
            }
            .title-text {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
                color: #f8fafc;
            }
            .sub-text {
                color: #cbd5e1;
                margin-bottom: 1.1rem;
            }
            .section-title {
                margin-top: 0.5rem;
                margin-bottom: 0.45rem;
                color: #dbeafe;
                font-weight: 600;
            }
            .stButton button {
                border-radius: 10px;
                font-weight: 600;
                padding: 0.45rem 1rem;
            }
            [data-testid="stDownloadButton"] button {
                color: #ffffff !important;
                background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
                border: 1px solid #60a5fa !important;
                border-radius: 10px !important;
                font-weight: 700 !important;
            }
            [data-testid="stDownloadButton"] button:hover {
                filter: brightness(1.08);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def validate_pump(air_temp, process_temp, rpm, torque, tool_wear):
    errors = []
    warnings = []

    if process_temp <= air_temp:
        errors.append("Process temperature must be greater than air temperature.")
    if rpm <= 0:
        errors.append("RPM must be greater than zero.")
    if torque <= 0:
        errors.append("Torque must be positive.")
    if tool_wear < 0:
        errors.append("Operation time cannot be negative.")
    if air_temp < 250 or air_temp > 350:
        warnings.append("Air temperature is outside typical operating range.")
    if rpm > 4000:
        warnings.append("RPM unusually high.")

    return errors, warnings


def validate_hx(hot_in, hot_out, cold_out, cold_flow):
    errors = []

    if hot_in <= hot_out:
        errors.append("Hot inlet temperature must be greater than hot outlet temperature.")
    if hot_out <= cold_out:
        errors.append("Hot outlet temperature must be greater than cold outlet temperature.")
    if hot_in <= 0 or hot_out <= 0 or cold_out <= 0:
        errors.append("All temperatures must be positive.")
    if cold_flow <= 0:
        errors.append("Mass flow rate must be positive.")

    return errors


def validate_gas_turbine(temperature, rpm, torque, vibration, power, fuel, air_pressure, exhaust, oil_temp):
    errors = []
    warnings = []

    if temperature < 500:
        warnings.append("Turbine temperature is below the normal operating range.")
    if rpm <= 0:
        errors.append("RPM must be greater than zero.")
    elif rpm > 25000:
        warnings.append("RPM is unusually high.")
    if torque <= 0:
        errors.append("Torque must be positive.")
    if vibration < 0:
        errors.append("Vibration cannot be negative.")
    if power <= 0:
        errors.append("Power output must be positive.")
    if fuel <= 0:
        errors.append("Fuel flow must be positive.")
    if air_pressure <= 0:
        errors.append("Air pressure must be positive.")
    if exhaust < 300:
        warnings.append("Exhaust temperature is unusually low.")
    if oil_temp <= 0:
        errors.append("Oil temperature must be positive.")

    return errors, warnings


def get_risk_level(probability):
    if probability < 30:
        return "Low", "#16a34a"
    if probability < 70:
        return "Medium", "#f59e0b"
    return "High", "#ef4444"


def calculate_health_score(probability):
    return round(max(0, min(100, 100 - probability)), 2)


def get_health_color(health_score):
    if health_score >= 70:
        return "#16a34a"
    if health_score >= 40:
        return "#f59e0b"
    return "#ef4444"


def risk_score(value, warning, critical, higher_is_bad=True):
    if higher_is_bad:
        if value < warning:
            return 0
        if value >= critical:
            return 100
        return round(40 + 60 * ((value - warning) / (critical - warning)), 1)

    if value > warning:
        return 0
    if value <= critical:
        return 100
    return round(40 + 60 * ((warning - value) / (warning - critical)), 1)


def input_value(inputs, *names, default=0):
    for name in names:
        if name in inputs:
            return inputs[name]

    def normalize(text):
        return str(text).replace("℃", "C").replace("°C", "C").replace("?C", "C")

    normalized_inputs = {normalize(key): value for key, value in inputs.items()}
    for name in names:
        normalized_name = normalize(name)
        if normalized_name in normalized_inputs:
            return normalized_inputs[normalized_name]

    return default


def build_risk_breakdown(appliance, inputs):
    if appliance == "Pump":
        values = [
            ("Air Temperature", risk_score(inputs["Air Temperature (K)"], 330, 350)),
            ("Process Temperature", risk_score(inputs["Process Temperature (K)"], 335, 360)),
            ("Rotational Speed", risk_score(inputs["Rotational Speed (rpm)"], 2800, 4000)),
            ("Torque", risk_score(inputs["Torque (Nm)"], 90, 120)),
            ("Operation Time", risk_score(inputs["Operation Time (min)"], 180, 250)),
            (
                "Temperature Gap",
                risk_score(
                    inputs["Process Temperature (K)"] - inputs["Air Temperature (K)"],
                    30,
                    50,
                ),
            ),
        ]
    elif appliance == "Heat Exchanger":
        values = [
            ("Hot Inlet Temperature", risk_score(inputs["Hot Inlet Temperature (K)"], 420, 450)),
            ("Hot Outlet Temperature", risk_score(inputs["Hot Outlet Temperature (K)"], 385, 410)),
            ("Cold Outlet Temperature", risk_score(inputs["Cold Outlet Temperature (K)"], 355, 380)),
            ("Cold Inlet Mass Flow", risk_score(inputs["Cold Inlet Mass Flow (kg/s)"], 4.0, 5.0)),
            (
                "Thermal Gradient",
                risk_score(
                    inputs["Hot Inlet Temperature (K)"] - inputs["Cold Outlet Temperature (K)"],
                    75,
                    190,
                ),
            ),
        ]
    elif appliance == "Boiler":
        values = [
            (
                "Flue Gas Temperature",
                risk_score(input_value(inputs, "Flue gas temperature (°C)", "Flue gas temperature (℃)"), 126, 140),
            ),
            ("Oxygen Level", risk_score(inputs["Boiler oxygen level (%)"], 5.0, 8.0)),
            ("CO", risk_score(inputs["CO (mg/m3)"], 450, 700)),
            ("Nox", risk_score(inputs["Nox (mg/m3)"], 230, 300)),
            ("SO2", risk_score(inputs["SO2 (mg/m3)"], 500, 650)),
            ("Dust", risk_score(inputs["Dust (mg/m3)"], 25, 40)),
            ("Boiler Efficiency", risk_score(inputs["Boiler Efficiency (%)"], 93.55, 92.0, higher_is_bad=False)),
            ("Gross Load", risk_score(inputs["Gross Load (MW)"], 292, 285, higher_is_bad=False)),
            ("Coal Flow", risk_score(inputs["Coal Flow (t/h)"], 172, 180)),
        ]
    else:
        values = [
            ("Turbine Temperature", risk_score(input_value(inputs, "Temperature (°C)", "Temperature (℃)"), 1050, 1200)),
            ("RPM", risk_score(inputs["RPM"], 25000, 30000)),
            ("Vibration", risk_score(inputs["Vibrations (mm/s)"], 3, 5)),
            ("Air Pressure", risk_score(inputs["Air Pressure (kPa)"], 105, 80, higher_is_bad=False)),
            (
                "Exhaust Temperature",
                risk_score(
                    input_value(inputs, "Exhaust Gas Temperature (°C)", "Exhaust Gas Temperature (℃)"),
                    560,
                    700,
                ),
            ),
            ("Oil Temperature", risk_score(input_value(inputs, "Oil Temperature (°C)", "Oil Temperature (℃)"), 140, 150)),
        ]

    return [
        {
            "Sensor": sensor,
            "Risk Score": float(score),
            "Severity": "High" if score >= 70 else "Medium" if score >= 40 else "Low",
        }
        for sensor, score in values
    ]


def pump_risk_flags(air_temp, process_temp, rpm, torque, tool_wear):
    flags = []
    if process_temp >= 335:
        flags.append(f"Process Temperature is high ({process_temp:.1f} K).")
    if air_temp >= 330:
        flags.append(f"Air Temperature is high ({air_temp:.1f} K).")
    if rpm >= 2800:
        flags.append(f"Rotational Speed is high ({rpm:.0f} rpm).")
    if torque >= 90:
        flags.append(f"Torque is high ({torque:.1f} Nm).")
    if tool_wear >= 180:
        flags.append(f"Operation Time is high ({tool_wear:.0f} min).")
    if (process_temp - air_temp) >= 30:
        flags.append(
            f"Temperature gap is high (Process-Air = {process_temp - air_temp:.1f} K)."
        )
    return flags


def hx_risk_flags(hot_in, hot_out, cold_out, cold_flow):
    flags = []
    if hot_in >= 420:
        flags.append(f"Hot Inlet Temperature is high ({hot_in:.1f} K).")
    if hot_out >= 385:
        flags.append(f"Hot Outlet Temperature is high ({hot_out:.1f} K).")
    if cold_out >= 355:
        flags.append(f"Cold Outlet Temperature is high ({cold_out:.1f} K).")
    if cold_flow >= 4.0:
        flags.append(f"Cold Inlet Mass Flow is high ({cold_flow:.2f} kg/s).")
    if (hot_in - cold_out) >= 75:
        flags.append(
            f"Thermal gradient is high (Hot Inlet - Cold Outlet = {hot_in - cold_out:.1f} K)."
        )
    return flags


def hx_physics_required(hot_in, hot_out, cold_out, cold_flow):
    risk_count = 0

    if hot_in >= 420:
        risk_count += 1
    if hot_out >= 385:
        risk_count += 1
    if cold_out >= 355:
        risk_count += 1
    if cold_flow >= 4.0:
        risk_count += 1
    if (hot_in - cold_out) >= 75:
        risk_count += 1

    critical_required = (
        hot_in >= 500
        or hot_out >= 450
        or cold_out >= 420
        or (hot_in - cold_out) >= 190
    )
    return critical_required or risk_count >= 3


def boiler_risk_flags(flue_temp, boiler_o2, co, nox, so2, dust, eff, gross_load, coal_flow):
    flags = []
    if flue_temp >= 126:
        flags.append(f"Flue gas temperature is high ({flue_temp:.1f} °C).")
    if boiler_o2 >= 5.0:
        flags.append(f"Boiler oxygen level is high ({boiler_o2:.2f}%).")
    if co >= 450:
        flags.append(f"CO is high ({co:.1f} mg/m3).")
    if nox >= 230:
        flags.append(f"Nox is high ({nox:.1f} mg/m3).")
    if so2 >= 500:
        flags.append(f"SO2 is high ({so2:.1f} mg/m3).")
    if dust >= 25:
        flags.append(f"Dust is high ({dust:.1f} mg/m3).")
    if eff <= 93.55:
        flags.append(f"Boiler efficiency is low ({eff:.2f}%).")
    if gross_load <= 292:
        flags.append(f"Gross load is low ({gross_load:.1f} MW).")
    if coal_flow >= 172:
        flags.append(f"Coal flow is high ({coal_flow:.1f} t/h).")
    return flags


def gas_turbine_physics(temperature, rpm, vibration, air_pressure, exhaust, oil_temp):
    risk_reasons = []
    critical_reasons = []

    if temperature > 1200:
        risk_reasons.append("turbine temperature is high")
        critical_reasons.append("turbine temperature is critically high")
    elif temperature > 1050:
        risk_reasons.append("turbine temperature is high")

    if rpm > 25000:
        critical_reasons.append("rotational speed is unusually high")
    if vibration > 3:
        risk_reasons.append("vibration is excessive")
    if vibration > 5:
        critical_reasons.append("vibration is critically high")
    if air_pressure < 105:
        risk_reasons.append("compressor air pressure is too low")
    if air_pressure < 80:
        critical_reasons.append("compressor air pressure is critically low")
    if exhaust > 560:
        risk_reasons.append("exhaust gas temperature is high")
    if exhaust > 700:
        critical_reasons.append("exhaust gas temperature is critically high")
    if oil_temp > 140:
        risk_reasons.append("oil temperature is high")
    if oil_temp > 150:
        critical_reasons.append("oil temperature is critically high")

    physics_required = len(risk_reasons) >= 2
    critical_required = len(critical_reasons) >= 1
    reasons = critical_reasons or risk_reasons

    return physics_required or critical_required, reasons


def gas_turbine_risk_flags(temperature, rpm, vibration, air_pressure, exhaust, oil_temp, probability):
    flags = []
    physics_required, reasons = gas_turbine_physics(
        temperature,
        rpm,
        vibration,
        air_pressure,
        exhaust,
        oil_temp,
    )

    if physics_required and probability < 30 and reasons:
        flags.append("Maintenance is required because " + ", ".join(reasons) + ".")
    elif reasons:
        flags.extend(reason.capitalize() + "." for reason in reasons)

    return flags


def build_theoretical_suggestions(appliance, inputs):
    causes = []
    actions = []

    if appliance == "Pump":
        if inputs["Process Temperature (K)"] >= 335:
            causes.append("High process temperature can accelerate seal and bearing degradation.")
            actions.append("Check cooling performance and reduce thermal load.")
        if inputs["Rotational Speed (rpm)"] >= 2800:
            causes.append("High rotational speed increases vibration and fatigue stress.")
            actions.append("Inspect shaft alignment and verify speed controller settings.")
        if inputs["Torque (Nm)"] >= 90:
            causes.append("High torque indicates mechanical overload or friction rise.")
            actions.append("Inspect impeller, bearings, and lubrication condition.")
        if inputs["Operation Time (min)"] >= 180:
            causes.append("Long operation time can increase component wear and reduce process stability.")
            actions.append("Schedule inspection or maintenance for components with extended runtime.")
    elif appliance == "Heat Exchanger":
        if inputs["Hot Inlet Temperature (K)"] >= 420:
            causes.append("Very high hot inlet temperature can increase thermal stress.")
            actions.append("Check heat source control and insulation performance.")
        if inputs["Cold Outlet Temperature (K)"] >= 355:
            causes.append("High cold outlet temperature may indicate reduced heat transfer efficiency.")
            actions.append("Inspect for fouling/scaling and clean exchanger surfaces.")
        if inputs["Cold Inlet Mass Flow (kg/s)"] >= 4.0:
            causes.append("High flow may cause pressure drop and unstable thermal behavior.")
            actions.append("Validate pump setpoint and control valve operation.")
        if (inputs["Hot Inlet Temperature (K)"] - inputs["Cold Outlet Temperature (K)"]) >= 75:
            causes.append("Large thermal gradient can increase material fatigue risk.")
            actions.append("Control thermal ramp rates and monitor expansion stress.")

    if appliance == "Boiler":
        flue_temp = inputs.get("Flue gas temperature (°C)", inputs.get("Flue gas temperature (℃)", 0))

        if flue_temp >= 126:
            causes.append("High flue gas temperature can indicate fouling/soot buildup and poor heat transfer.")
            actions.append("Inspect soot blowers, clean heat transfer surfaces, and verify draft settings.")
        if inputs["Boiler oxygen level (%)"] >= 5.0:
            causes.append("High oxygen level can indicate excess air, reducing efficiency and increasing heat loss.")
            actions.append("Tune combustion (air-fuel ratio) and check air leakage.")
        if inputs["CO (mg/m3)"] >= 450:
            causes.append("High CO indicates incomplete combustion.")
            actions.append("Check burner condition, mixing, and fuel quality; tune combustion.")
        if inputs["Boiler Efficiency (%)"] <= 93.55:
            causes.append("Low efficiency indicates higher fuel usage for same output.")
            actions.append("Check economizer, air preheater leakage, and overall heat balance.")

    if appliance == "Gas Turbine":
        temperature = inputs.get("Temperature (°C)", inputs.get("Temperature (℃)", 0))
        exhaust = inputs.get("Exhaust Gas Temperature (°C)", inputs.get("Exhaust Gas Temperature (℃)", 0))
        oil_temp = inputs.get("Oil Temperature (°C)", inputs.get("Oil Temperature (℃)", 0))

        if temperature >= 1050:
            causes.append("High turbine temperature increases thermal stress on hot-section components.")
            actions.append("Review turbine inlet temperature trend, combustion tuning, and cooling airflow.")
        if inputs["Vibrations (mm/s)"] >= 3:
            causes.append("High vibration can indicate imbalance, bearing wear, or shaft alignment issues.")
            actions.append("Inspect bearings, coupling alignment, and vibration trend history.")
        if inputs["Air Pressure (kPa)"] < 105:
            causes.append("Low compressor air pressure can reduce combustion stability and turbine efficiency.")
            actions.append("Check compressor health, inlet filters, and pressure sensors.")
        if exhaust >= 560:
            causes.append("High exhaust gas temperature can indicate combustion or cooling problems.")
            actions.append("Inspect combustor condition, fuel-air ratio, and exhaust temperature spread.")
        if oil_temp >= 140:
            causes.append("High oil temperature can reduce lubrication quality and bearing protection.")
            actions.append("Check oil cooler performance, oil level, and lubrication circuit condition.")

    if not causes:
        causes.append("No dominant high-risk variable detected; risk is likely from combined conditions.")
        actions.append("Continue monitoring trend data and inspect at routine intervals.")

    return causes, actions


def build_report_data(appliance, prediction, probability, risk_text, flags, inputs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_text = "Maintenance Required" if prediction == 1 else "System Healthy"
    causes, actions = build_theoretical_suggestions(appliance, inputs)
    health_score = calculate_health_score(probability)
    risk_breakdown = build_risk_breakdown(appliance, inputs)
    return {
        "timestamp": timestamp,
        "appliance": appliance,
        "result": result_text,
        "failure_probability_percent": round(probability, 2),
        "health_score": health_score,
        "risk_level": risk_text,
        "inputs": inputs,
        "flags": flags,
        "risk_breakdown": risk_breakdown,
        "possible_causes": causes,
        "recommended_actions": actions,
    }


def generate_csv_report(report):
    row = {
        "Timestamp": report["timestamp"],
        "Appliance": report["appliance"],
        "Result": report["result"],
        "Failure Probability (%)": report["failure_probability_percent"],
        "Health Score": report["health_score"],
        "Risk Level": report["risk_level"],
        "Input Risk Flags": " | ".join(report["flags"]) if report["flags"] else "None",
        "Risk Breakdown": " | ".join(
            f"{item['Sensor']}: {item['Risk Score']} ({item['Severity']})"
            for item in report["risk_breakdown"]
        ),
        "Possible Causes": " | ".join(report["possible_causes"]),
        "Recommended Actions": " | ".join(report["recommended_actions"]),
    }
    row.update(report["inputs"])
    df = pd.DataFrame([row])
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf_report(report):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def clean_text(text):
        return (
            str(text)
            .replace("℃", "deg C")
            .replace("°", "deg ")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )

    def write_line(text, bold=False, size=9, height=5):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Arial", "B" if bold else "", size)
        pdf.multi_cell(usable_width, height, clean_text(text))

    def bar_color(score, health=False):
        if health:
            if score >= 70:
                return 22, 163, 74
            if score >= 40:
                return 245, 158, 11
            return 239, 68, 68
        if score >= 70:
            return 239, 68, 68
        if score >= 40:
            return 245, 158, 11
        return 22, 163, 74

    def draw_bar(label, value, x, y, width, height=5, health=False):
        pdf.set_xy(x, y)
        pdf.set_font("Arial", "", 8)
        pdf.cell(50, height, clean_text(label))
        bar_x = x + 52
        pdf.set_fill_color(229, 231, 235)
        pdf.rect(bar_x, y + 0.8, width, height - 1.6, "F")
        pdf.set_fill_color(*bar_color(value, health=health))
        pdf.rect(bar_x, y + 0.8, width * max(0, min(100, value)) / 100, height - 1.6, "F")
        pdf.set_xy(bar_x + width + 3, y)
        pdf.cell(18, height, f"{value:.1f}")

    def write_box(title, items, x, y, width, max_items=4):
        pdf.set_xy(x, y)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(width, 5, clean_text(title))
        y += 6
        pdf.set_font("Arial", "", 8.2)
        for item in items[:max_items]:
            pdf.set_xy(x, y)
            pdf.multi_cell(width, 4.3, clean_text(f"- {item}"))
            y = pdf.get_y()
        return y

    write_line("Predictive Maintenance Report", bold=True, size=15)
    write_line(f"{report['timestamp']} | {report['appliance']}", size=9)
    pdf.ln(1)

    write_line("Decision Summary", bold=True, size=11)
    write_line(f"Status: {report['result']}", size=9)
    write_line(f"Failure Probability: {report['failure_probability_percent']}% | Risk Level: {report['risk_level']}", size=9)
    draw_bar("Health Score", report["health_score"], pdf.l_margin, pdf.get_y() + 1, 112, health=True)
    pdf.ln(10)

    top_risks = sorted(report["risk_breakdown"], key=lambda item: item["Risk Score"], reverse=True)[:5]
    has_risk_breakdown = any(item["Risk Score"] > 0 for item in top_risks)

    if has_risk_breakdown:
        write_line("Risk Breakdown", bold=True, size=11)
        y = pdf.get_y() + 1
        for item in top_risks:
            draw_bar(item["Sensor"], item["Risk Score"], pdf.l_margin, y, 112)
            y += 6
        pdf.set_y(y + 2)
    else:
        write_line("Operating Condition", bold=True, size=11)
        condition_text = (
            "One or more monitored inputs are outside the normal risk range."
            if report["flags"]
            else "All monitored inputs are within the normal risk range."
        )
        write_line(condition_text, size=9)

    write_line("Input Risk Flags", bold=True, size=11)
    if report["flags"]:
        for item in report["flags"][:4]:
            write_line(f"- {item}", size=8.5, height=4.6)
    else:
        write_line("- No individual input is in the high-risk range.", size=8.5, height=4.6)

    pdf.ln(1)
    write_line("Important Inputs", bold=True, size=11)
    input_items = list(report["inputs"].items())
    pdf.set_font("Arial", "", 8.2)
    for key, value in input_items:
        write_line(f"- {key}: {value}", size=8.5, height=4.6)
    pdf.ln(1)

    pdf.ln(1)
    section_y = write_box("Probable Causes", report["possible_causes"], pdf.l_margin, pdf.get_y(), usable_width, max_items=4)
    pdf.set_y(section_y + 2)
    section_y = write_box("Recommended Actions", report["recommended_actions"], pdf.l_margin, pdf.get_y(), usable_width, max_items=4)
    pdf.set_y(section_y + 2)

    conclusion = (
        "Maintenance attention is recommended based on the model output and sensor condition."
        if report["result"] == "Maintenance Required"
        else "Machine condition is healthy; continue regular monitoring and routine preventive maintenance."
    )
    write_line("Conclusion", bold=True, size=11)
    write_line(conclusion, size=8.5, height=4.5)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        output = output.encode("latin-1")
    return bytes(output)


def render_health_score(health_score):
    health_color = get_health_color(health_score)
    st.markdown("**Health Score**")
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem;">
                <span style="font-weight:700; color:#e2e8f0;">Overall Machine Health</span>
                <span style="font-size:1.6rem; font-weight:800; color:#ffffff;">{health_score:.2f}/100</span>
            </div>
            <div style="height:14px; background:rgba(148,163,184,0.25); border-radius:999px; overflow:hidden;">
                <div style="width:{health_score}%; height:100%; background:{health_color}; border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_breakdown_chart(risk_breakdown):
    st.markdown("**Risk Breakdown**")
    chart_data = pd.DataFrame(risk_breakdown).sort_values("Risk Score", ascending=False)

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Risk Score:Q", scale=alt.Scale(domain=[0, 100]), title="Risk Score"),
            y=alt.Y("Sensor:N", sort="-x", title=None),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=["Low", "Medium", "High"],
                    range=["#16a34a", "#f59e0b", "#ef4444"],
                ),
                legend=None,
            ),
            tooltip=["Sensor", "Risk Score", "Severity"],
        )
        .properties(height=max(190, 28 * len(chart_data)))
    )
    st.altair_chart(chart, use_container_width=True)


def show_result(appliance, prediction, probability, flags, inputs):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Maintenance Required")
    else:
        st.success("System Healthy")
    risk_text, risk_color = get_risk_level(probability)
    risk_breakdown = build_risk_breakdown(appliance, inputs)
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            margin: 0.35rem 0 0.55rem 0;
            background:{risk_color};
            color:#ffffff;
            font-weight:700;
            border-radius:999px;
            padding:0.28rem 0.75rem;
            font-size:0.86rem;
            letter-spacing:0.2px;">
            Risk Level: {risk_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Failure Probability", f"{probability:.2f}%")
    health_score = calculate_health_score(probability)
    render_health_score(health_score)

    if prediction == 1:
        render_risk_breakdown_chart(risk_breakdown)

    st.markdown("**Input Risk Flags**")
    if flags:
        for item in flags:
            st.warning(item)
    else:
        st.success("No individual input is in the high-risk range.")

    report = build_report_data(
        appliance=appliance,
        prediction=prediction,
        probability=probability,
        risk_text=risk_text,
        flags=flags,
        inputs=inputs,
    )

    st.markdown("**Download Report**")
    csv_bytes = generate_csv_report(report)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"maintenance_report_{report['timestamp'].replace(':', '-').replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        try:
            pdf_bytes = generate_pdf_report(report)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"maintenance_report_{report['timestamp'].replace(':', '-').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")
            st.info(
                f"Install in current runtime with: `{sys.executable} -m pip install fpdf2`"
            )


def main():
    style_app()

    st.markdown('<div class="title-text">Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">Input real-time sensor values to estimate failure risk.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Machine Configuration</div>', unsafe_allow_html=True)
    appliance = st.selectbox("Select Appliance", ["Pump", "Heat Exchanger", "Boiler", "Gas Turbine"])

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Sensor Inputs</div>', unsafe_allow_html=True)

        if appliance == "Pump":
            col1, col2 = st.columns(2)
            with col1:
                air_temp = st.number_input("Air Temperature (K)", value=300.0, step=0.1)
                process_temp = st.number_input("Process Temperature (K)", value=310.0, step=0.1)
                rpm = st.number_input("Rotational Speed (rpm)", value=1500.0, step=1.0)
            with col2:
                torque = st.number_input("Torque (Nm)", value=45.0, step=0.1)
                tool_wear = st.number_input("Operation Time (min)", value=20.0, step=1.0)
        elif appliance == "Heat Exchanger":
            col1, col2 = st.columns(2)
            with col1:
                hot_in = st.number_input("Hot Inlet Temperature (K)", value=390.0, step=0.1)
                hot_out = st.number_input("Hot Outlet Temperature (K)", value=360.0, step=0.1)
            with col2:
                cold_out = st.number_input("Cold Outlet Temperature (K)", value=330.0, step=0.1)
                cold_flow = st.number_input("Cold Inlet Mass Flow (kg/s)", value=2.0, step=0.1)
        elif appliance == "Boiler":
            col1, col2 = st.columns(2)
            with col1:
                flue_temp = st.number_input("Flue gas temperature (°C)", value=123.0, step=0.1)
                boiler_o2 = st.number_input("Boiler oxygen level (%)", value=4.7, step=0.01)
                co = st.number_input("CO (mg/m3)", value=260.0, step=1.0)
                nox = st.number_input("Nox (mg/m3)", value=195.0, step=1.0)
                so2 = st.number_input("SO2 (mg/m3)", value=438.0, step=1.0)
            with col2:
                dust = st.number_input("Dust (mg/m3)", value=14.5, step=0.1)
                eff = st.number_input("Boiler Efficiency (%)", value=93.67, step=0.01)
                gross_load = st.number_input("Gross Load (MW)", value=296.0, step=0.1)
                coal_flow = st.number_input("Coal Flow (t/h)", value=167.0, step=0.1)
        else:
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input("Temperature (°C)", value=951.0, step=0.1)
                rpm_gas = st.number_input("RPM", value=16203.0, step=1.0)
                torque_gas = st.number_input("Torque (Nm)", value=3566.0, step=0.1)
                vibration = st.number_input("Vibrations (mm/s)", value=1.72, step=0.01)
                power = st.number_input("Power Output (MW)", value=88.0, step=0.1)
            with col2:
                fuel = st.number_input("Fuel Flow Rate (kg/s)", value=2.77, step=0.01)
                air_pressure = st.number_input("Air Pressure (kPa)", value=159.0, step=0.1)
                exhaust = st.number_input("Exhaust Gas Temperature (°C)", value=454.0, step=0.1)
                oil_temp = st.number_input("Oil Temperature (°C)", value=127.0, step=0.1)

        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if submitted and appliance == "Pump":
        errors, warnings = validate_pump(air_temp, process_temp, rpm, torque, tool_wear)
        if errors:
            for error in errors:
                st.warning(error)
        else:
            for warning in warnings:
                st.info(warning)

            model, scaler = load_pump_assets()
            data = pd.DataFrame(
                [[air_temp, process_temp, rpm, torque, tool_wear]],
                columns=PUMP_FEATURES,
            )
            data_scaled = scaler.transform(data)
            prediction = int(model.predict(data_scaled)[0])
            probability = float(model.predict_proba(data_scaled)[0][1] * 100)
            flags = pump_risk_flags(air_temp, process_temp, rpm, torque, tool_wear)
            inputs = {
                "Air Temperature (K)": air_temp,
                "Process Temperature (K)": process_temp,
                "Rotational Speed (rpm)": rpm,
                "Torque (Nm)": torque,
                "Operation Time (min)": tool_wear,
            }
            show_result("Pump", prediction, probability, flags, inputs)

    if submitted and appliance == "Heat Exchanger":
        errors = validate_hx(hot_in, hot_out, cold_out, cold_flow)
        if errors:
            for error in errors:
                st.warning(error)
        else:
            model, scaler = load_hx_assets()
            data = pd.DataFrame(
                [[hot_in, hot_out, cold_out, cold_flow]],
                columns=HX_FEATURES,
            )
            data_scaled = scaler.transform(data)
            prediction = int(model.predict(data_scaled)[0])
            probability = float(model.predict_proba(data_scaled)[0][1] * 100)
            flags = hx_risk_flags(hot_in, hot_out, cold_out, cold_flow)
            if hx_physics_required(hot_in, hot_out, cold_out, cold_flow):
                prediction = 1
            inputs = {
                "Hot Inlet Temperature (K)": hot_in,
                "Hot Outlet Temperature (K)": hot_out,
                "Cold Outlet Temperature (K)": cold_out,
                "Cold Inlet Mass Flow (kg/s)": cold_flow,
            }
            show_result("Heat Exchanger", prediction, probability, flags, inputs)

    if submitted and appliance == "Boiler":
        if flue_temp <= 0:
            st.warning("Flue gas temperature must be positive.")
        elif boiler_o2 < 0 or boiler_o2 > 25:
            st.warning("Boiler oxygen level must be between 0 and 25.")
        elif eff <= 0 or eff > 100:
            st.warning("Boiler efficiency must be between 0 and 100.")
        elif gross_load <= 0:
            st.warning("Gross load must be positive.")
        elif coal_flow <= 0:
            st.warning("Coal flow must be positive.")
        else:
            model, scaler = load_boiler_assets()
            data = pd.DataFrame(
                [[flue_temp, boiler_o2, co, nox, so2, dust, eff, gross_load, coal_flow]],
                columns=BOILER_FEATURES,
            )
            data_scaled = scaler.transform(data)
            prediction = int(model.predict(data_scaled)[0])
            probability = float(model.predict_proba(data_scaled)[0][1] * 100)
            flags = boiler_risk_flags(flue_temp, boiler_o2, co, nox, so2, dust, eff, gross_load, coal_flow)
            inputs = {
                "Flue gas temperature (°C)": flue_temp,
                "Boiler oxygen level (%)": boiler_o2,
                "CO (mg/m3)": co,
                "Nox (mg/m3)": nox,
                "SO2 (mg/m3)": so2,
                "Dust (mg/m3)": dust,
                "Boiler Efficiency (%)": eff,
                "Gross Load (MW)": gross_load,
                "Coal Flow (t/h)": coal_flow,
            }
            show_result("Boiler", prediction, probability, flags, inputs)

    if submitted and appliance == "Gas Turbine":
        errors, warnings = validate_gas_turbine(
            temperature,
            rpm_gas,
            torque_gas,
            vibration,
            power,
            fuel,
            air_pressure,
            exhaust,
            oil_temp,
        )
        if errors:
            for error in errors:
                st.warning(error)
        else:
            for warning in warnings:
                st.info(warning)

            thermal_stress = temperature * exhaust
            mechanical_stress = rpm_gas * vibration
            combustion_index = fuel / (air_pressure + 1)
            oil_risk = oil_temp * vibration

            model, scaler = load_gas_turbine_assets()
            data = pd.DataFrame(
                [[
                    temperature,
                    rpm_gas,
                    torque_gas,
                    vibration,
                    power,
                    fuel,
                    air_pressure,
                    exhaust,
                    oil_temp,
                    thermal_stress,
                    mechanical_stress,
                    combustion_index,
                    oil_risk,
                ]],
                columns=GAS_TURBINE_FEATURES,
            )
            data_scaled = scaler.transform(data)
            prediction = int(model.predict(data_scaled)[0])
            probability = float(model.predict_proba(data_scaled)[0][1] * 100)
            physics_required, _ = gas_turbine_physics(
                temperature,
                rpm_gas,
                vibration,
                air_pressure,
                exhaust,
                oil_temp,
            )
            if physics_required:
                prediction = 1

            flags = gas_turbine_risk_flags(
                temperature,
                rpm_gas,
                vibration,
                air_pressure,
                exhaust,
                oil_temp,
                probability,
            )
            inputs = {
                "Temperature (°C)": temperature,
                "RPM": rpm_gas,
                "Torque (Nm)": torque_gas,
                "Vibrations (mm/s)": vibration,
                "Power Output (MW)": power,
                "Fuel Flow Rate (kg/s)": fuel,
                "Air Pressure (kPa)": air_pressure,
                "Exhaust Gas Temperature (°C)": exhaust,
                "Oil Temperature (°C)": oil_temp,
            }
            show_result("Gas Turbine", prediction, probability, flags, inputs)


if __name__ == "__main__":
    main()
