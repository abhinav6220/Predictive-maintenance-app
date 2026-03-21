from pathlib import Path
from datetime import datetime
import sys

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
        errors.append("Tool wear cannot be negative.")
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


def get_risk_level(probability):
    if probability < 30:
        return "Low", "#16a34a"
    if probability < 70:
        return "Medium", "#f59e0b"
    return "High", "#ef4444"


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
        flags.append(f"Tool Wear is high ({tool_wear:.0f} min).")
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
        if inputs["Tool Wear (min)"] >= 180:
            causes.append("High wear level can reduce process stability and efficiency.")
            actions.append("Schedule replacement/maintenance for worn components.")
    else:
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

    if not causes:
        causes.append("No dominant high-risk variable detected; risk is likely from combined conditions.")
        actions.append("Continue monitoring trend data and inspect at routine intervals.")

    return causes, actions


def build_report_data(appliance, prediction, probability, risk_text, flags, inputs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_text = "Maintenance Required" if prediction == 1 else "System Healthy"
    causes, actions = build_theoretical_suggestions(appliance, inputs)
    return {
        "timestamp": timestamp,
        "appliance": appliance,
        "result": result_text,
        "failure_probability_percent": round(probability, 2),
        "risk_level": risk_text,
        "inputs": inputs,
        "flags": flags,
        "possible_causes": causes,
        "recommended_actions": actions,
    }


def generate_csv_report(report):
    row = {
        "Timestamp": report["timestamp"],
        "Appliance": report["appliance"],
        "Result": report["result"],
        "Failure Probability (%)": report["failure_probability_percent"],
        "Risk Level": report["risk_level"],
        "Input Risk Flags": " | ".join(report["flags"]) if report["flags"] else "None",
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
    pdf.set_auto_page_break(auto=True, margin=12)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def write_line(text, bold=False, size=11):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Arial", "B" if bold else "", size)
        pdf.multi_cell(usable_width, 7, str(text))

    write_line("Predictive Maintenance Report", bold=True, size=15)
    write_line(f"Timestamp: {report['timestamp']}")
    write_line(f"Appliance: {report['appliance']}")
    write_line(f"Result: {report['result']}")
    write_line(f"Failure Probability: {report['failure_probability_percent']}%")
    write_line(f"Risk Level: {report['risk_level']}")
    pdf.ln(2)

    write_line("Input Values", bold=True, size=12)
    for key, value in report["inputs"].items():
        write_line(f"- {key}: {value}")

    pdf.ln(1)
    write_line("Input Risk Flags", bold=True, size=12)
    if report["flags"]:
        for item in report["flags"]:
            write_line(f"- {item}")
    else:
        write_line("- No individual input is in the high-risk range.")

    pdf.ln(1)
    write_line("Possible Causes (Theoretical)", bold=True, size=12)
    for cause in report["possible_causes"]:
        write_line(f"- {cause}")

    pdf.ln(1)
    write_line("Recommended Actions", bold=True, size=12)
    for action in report["recommended_actions"]:
        write_line(f"- {action}")

    output = pdf.output(dest="S")
    if isinstance(output, str):
        output = output.encode("latin-1")
    return bytes(output)


def show_result(appliance, prediction, probability, flags, inputs):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Maintenance Required")
    else:
        st.success("System Healthy")
    risk_text, risk_color = get_risk_level(probability)
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
    appliance = st.selectbox("Select Appliance", ["Pump", "Heat Exchanger"])

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
                tool_wear = st.number_input("Tool Wear (min)", value=20.0, step=1.0)
        else:
            col1, col2 = st.columns(2)
            with col1:
                hot_in = st.number_input("Hot Inlet Temperature (K)", value=390.0, step=0.1)
                hot_out = st.number_input("Hot Outlet Temperature (K)", value=360.0, step=0.1)
            with col2:
                cold_out = st.number_input("Cold Outlet Temperature (K)", value=330.0, step=0.1)
                cold_flow = st.number_input("Cold Inlet Mass Flow (kg/s)", value=2.0, step=0.1)

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
                "Tool Wear (min)": tool_wear,
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
            inputs = {
                "Hot Inlet Temperature (K)": hot_in,
                "Hot Outlet Temperature (K)": hot_out,
                "Cold Outlet Temperature (K)": cold_out,
                "Cold Inlet Mass Flow (kg/s)": cold_flow,
            }
            show_result("Heat Exchanger", prediction, probability, flags, inputs)


if __name__ == "__main__":
    main()
