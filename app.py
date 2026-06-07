"""
CureOpt AI — Streamlit Dashboard
AI-Powered Cycle Time Optimization for Precast Yards
L&T CreaTech 2025
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLIMATIC_PROFILES, CURING_METHOD_LABELS, ADMIXTURE_LABELS,
    AUTOMATION_LABELS, DASHBOARD_CONFIG, BASELINE_CYCLE_TIME_HR,
    SAFETY_FACTOR, XGBOOST_MODEL_PATH
)

# ─── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_CONFIG["page_title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

# ─── Auto-train models if not present (first launch on cloud) ──────────
if not os.path.exists(XGBOOST_MODEL_PATH):
    with st.spinner("🔧 First launch — generating dataset & training models. This may take a few minutes..."):
        from data.generate_dataset import save_dataset
        from models.train_model import train_full_pipeline
        save_dataset()
        train_full_pipeline()

# ─── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero gradient header */
    .hero-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a237e 40%, #0d47a1 70%, #01579b 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(66, 165, 245, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.75);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* KPI Cards row */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 100%);
        border: 1px solid rgba(26, 35, 126, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(13, 71, 161, 0.12);
        border-color: rgba(26, 35, 126, 0.15);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #43a047;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(67, 160, 71, 0.15);
    }
    .rec-card h3 {
        color: #1b5e20;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .rec-metric {
        display: inline-block;
        background: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.3rem;
        border: 1px solid rgba(67, 160, 71, 0.2);
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .rec-metric .label {
        font-size: 0.7rem;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .rec-metric .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1b5e20;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a237e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e8eaf6;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a237e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500;
    }

    /* Scenario cards */
    .scenario-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }
    .scenario-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .scenario-rank {
        display: inline-block;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏗️ CureOpt AI</h1>
    <p>AI-Powered Cycle Time Optimization for Precast Yards · L&T CreaTech 2025</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar Inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.markdown("---")

    # Site Profile
    st.markdown("### 🌍 Site Profile")
    profile_options = list(CLIMATIC_PROFILES.keys()) + ["Custom"]
    selected_profile = st.selectbox("Climatic Zone", profile_options, index=0)

    if selected_profile != "Custom":
        profile = CLIMATIC_PROFILES[selected_profile]
        default_temp = profile["ambient_temp_C"]
        default_humidity = profile["humidity_pct"]
        st.info(f"📍 {profile['description']}")
    else:
        default_temp = 30.0
        default_humidity = 70.0

    st.markdown("---")

    # Environment
    st.markdown("### 🌡️ Environment")
    ambient_temp = st.slider("Ambient Temperature (°C)", 5.0, 45.0, default_temp, 0.5)
    humidity = st.slider("Relative Humidity (%)", 30.0, 95.0, default_humidity, 1.0)

    st.markdown("---")

    # Mix Design
    st.markdown("### 🧪 Mix Design")
    cement_pct = st.slider("Cement Content (%)", 35.0, 65.0, 50.0, 0.5)
    fly_ash_pct = st.slider("Fly Ash Content (%)", 0.0, 30.0, 10.0, 0.5)
    wc_ratio = st.slider("Water-Cement Ratio", 0.35, 0.55, 0.45, 0.01)

    st.markdown("---")

    # Curing
    st.markdown("### 🔥 Curing & Admixture")
    curing_labels = list(CURING_METHOD_LABELS.values())
    curing_method_str = st.radio("Curing Method", curing_labels, index=0)
    curing_method_int = {v: k for k, v in CURING_METHOD_LABELS.items()}[curing_method_str]

    admixture_labels = list(ADMIXTURE_LABELS.values())
    admixture_str = st.selectbox("Admixture Type", admixture_labels, index=0)
    admixture_int = {v: k for k, v in ADMIXTURE_LABELS.items()}[admixture_str]

    st.markdown("---")

    # Automation & Strength
    st.markdown("### 🏭 Operations")
    automation = st.select_slider(
        "Automation Level",
        options=[1, 2, 3],
        value=2,
        format_func=lambda x: AUTOMATION_LABELS[x]
    )
    required_mpa = st.number_input("Required Strength (MPa)", 15.0, 50.0, 25.0, 1.0)

    st.markdown("---")

    # Run button
    run_clicked = st.button("🚀 Run CureOpt AI", use_container_width=True, type="primary")


# ─── Main Panel ─────────────────────────────────────────────────────────

def run_analysis():
    """Execute the full CureOpt AI pipeline and render results."""

    from models.predict import predict_strength_curve, find_earliest_demould, get_shap_explanation
    from optimizer.cost_model import calculate_cost, calculate_baseline_cost
    from optimizer.ga_optimizer import optimize, run_scenario_comparison

    features = {
        "cement_pct": cement_pct,
        "fly_ash_pct": fly_ash_pct,
        "water_cement_ratio": wc_ratio,
        "curing_method": curing_method_int,
        "admixture_type": admixture_int,
        "ambient_temp_C": ambient_temp,
        "humidity_pct": humidity,
        "curing_duration_hr": 24.0,
        "automation_level": automation,
    }

    region = selected_profile if selected_profile != "Custom" else "Custom"

    # ─── Progress ───────────────────────────────────────────
    progress = st.progress(0, text="🔄 Initializing CureOpt AI engine...")
    time.sleep(0.3)

    # 1. Strength curve prediction
    progress.progress(15, text="📈 Predicting strength development curve...")
    hours = list(range(1, 37))
    curve = predict_strength_curve(features, hours)
    curve_hours = [c[0] for c in curve]
    curve_strengths = [c[1] for c in curve]

    # 2. Earliest demould
    progress.progress(30, text="🔍 Finding earliest safe de-mould time...")
    demould = find_earliest_demould(features, required_mpa, SAFETY_FACTOR)

    # 3. SHAP explanation
    progress.progress(45, text="🧠 Computing SHAP feature explanations...")
    shap_values = get_shap_explanation(features)

    # 4. Cost calculation
    progress.progress(55, text="💰 Computing cost breakdown...")
    cost_params = features.copy()
    cost_params["cycle_time_hr"] = demould["demould_hour"]
    cost_params["curing_method"] = curing_method_str
    cost_params["admixture_type"] = admixture_str
    cost = calculate_cost(cost_params, region=region, automation_level=automation)
    baseline_cost = calculate_baseline_cost(region=region, automation_level=automation)

    # 5. GA Optimization
    progress.progress(70, text="🧬 Running Genetic Algorithm optimization...")
    optimal = optimize(
        required_mpa=required_mpa,
        region=region,
        ambient_temp=ambient_temp,
        humidity=humidity,
        automation_level=automation,
        use_ga=True
    )

    # 6. Scenario comparison
    progress.progress(85, text="📊 Generating scenario comparisons...")
    scenarios = run_scenario_comparison(
        required_mpa=required_mpa,
        region=region,
        ambient_temp=ambient_temp,
        humidity=humidity,
        automation_level=automation,
        top_n=4
    )

    progress.progress(100, text="✅ Analysis complete!")
    time.sleep(0.5)
    progress.empty()

    # ─── KPI Cards ──────────────────────────────────────────
    time_reduction = (1 - demould["demould_hour"] / BASELINE_CYCLE_TIME_HR) * 100
    cost_index = cost["total"] / max(baseline_cost["total"], 1)
    mold_turns = 24.0 / max(demould["demould_hour"], 1)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{demould["demould_hour"]:.1f}h</div>
            <div class="kpi-label">De-mould Time</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{time_reduction:+.1f}%</div>
            <div class="kpi-label">Time Saved</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{demould["predicted_strength"]:.1f} MPa</div>
            <div class="kpi-label">Strength @ De-mould</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">₹{cost["total"]:,.0f}</div>
            <div class="kpi-label">Cycle Cost</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{mold_turns:.2f}×</div>
            <div class="kpi-label">Mold Turns/Day</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Charts Row 1: Strength Curve + Cost Breakdown ──────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">📈 Strength Development Curve</div>',
                    unsafe_allow_html=True)

        fig_curve = go.Figure()

        # Strength curve
        fig_curve.add_trace(go.Scatter(
            x=curve_hours, y=curve_strengths,
            mode="lines",
            name="Predicted Strength",
            line=dict(color="#1a73e8", width=3),
            fill="tozeroy",
            fillcolor="rgba(26, 115, 232, 0.08)",
        ))

        # Required strength line
        fig_curve.add_hline(
            y=required_mpa, line_dash="dash", line_color="#ea4335", line_width=2,
            annotation_text=f"Required: {required_mpa} MPa",
            annotation_position="top right",
            annotation_font_color="#ea4335",
        )

        # Safety threshold line
        safety_target = required_mpa * SAFETY_FACTOR
        fig_curve.add_hline(
            y=safety_target, line_dash="dot", line_color="#fbbc04", line_width=1.5,
            annotation_text=f"Safety Target: {safety_target:.1f} MPa (×{SAFETY_FACTOR})",
            annotation_position="bottom right",
            annotation_font_color="#fbbc04",
        )

        # De-mould marker
        fig_curve.add_vline(
            x=demould["demould_hour"], line_dash="dash", line_color="#34a853", line_width=2,
            annotation_text=f"De-mould: {demould['demould_hour']:.1f}h",
            annotation_position="top left",
            annotation_font_color="#34a853",
        )

        fig_curve.update_layout(
            template="plotly_white",
            xaxis_title="Time (hours)",
            yaxis_title="Compressive Strength (MPa)",
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💰 Cost Breakdown</div>',
                    unsafe_allow_html=True)

        cost_labels = ["Material", "Energy", "Labor", "Mold Occupancy"]
        cost_values = [cost["material"], cost["energy"], cost["labor"], cost["mold_occupancy"]]
        cost_colors = ["#1a73e8", "#ea4335", "#fbbc04", "#34a853"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=cost_labels,
            values=cost_values,
            hole=0.55,
            marker=dict(colors=cost_colors),
            textinfo="label+percent",
            textfont=dict(size=11, family="Inter"),
            hoverinfo="label+value+percent",
            hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
        )])

        fig_donut.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter"),
            showlegend=False,
            annotations=[dict(
                text=f"<b>₹{cost['total']:,.0f}</b>",
                x=0.5, y=0.5, font_size=18, font_family="Inter",
                font_color="#1a237e", showarrow=False
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ─── Charts Row 2: SHAP + Cycle Time Comparison ─────────
    col3, col4 = st.columns([2, 3])

    with col3:
        st.markdown('<div class="section-header">🧠 Feature Importance (SHAP)</div>',
                    unsafe_allow_html=True)

        shap_names = [s[0].replace("_", " ").title() for s in shap_values]
        shap_vals = [abs(s[1]) for s in shap_values]

        fig_shap = go.Figure(data=[go.Bar(
            y=shap_names[::-1],
            x=shap_vals[::-1],
            orientation="h",
            marker=dict(
                color=shap_vals[::-1],
                colorscale=[[0, "#e8eaf6"], [1, "#1a237e"]],
            ),
            text=[f"{v:.3f}" for v in shap_vals[::-1]],
            textposition="outside",
            textfont=dict(size=11, family="Inter"),
        )])

        fig_shap.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=120, r=60, t=20, b=30),
            font=dict(family="Inter"),
            xaxis_title="Impact on Prediction",
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">📊 Cycle Time Comparison</div>',
                    unsafe_allow_html=True)

        if scenarios:
            # Group by curing method
            sc_df = pd.DataFrame(scenarios[:12])  # up to 12 for comparison
            if not sc_df.empty and "curing_method" in sc_df.columns and "cycle_time_hr" in sc_df.columns:
                # Get best per curing method
                best_per_method = sc_df.groupby("curing_method").agg({
                    "cycle_time_hr": "min",
                    "predicted_strength": "max",
                }).reset_index()

                colors = {"Normal": "#1a73e8", "Steam": "#ea4335", "Heated Chamber": "#fbbc04"}

                fig_bar = go.Figure()
                for _, row in best_per_method.iterrows():
                    method = row["curing_method"]
                    ct = row["cycle_time_hr"]
                    reduction = (1 - ct / BASELINE_CYCLE_TIME_HR) * 100

                    fig_bar.add_trace(go.Bar(
                        x=[method],
                        y=[ct],
                        name=method,
                        marker_color=colors.get(method, "#5f6368"),
                        text=[f"{ct:.1f}h ({reduction:+.0f}%)"],
                        textposition="outside",
                        textfont=dict(size=12, family="Inter", color="#202124"),
                    ))

                # Baseline line
                fig_bar.add_hline(
                    y=BASELINE_CYCLE_TIME_HR, line_dash="dash",
                    line_color="#9e9e9e", line_width=1.5,
                    annotation_text=f"Baseline: {BASELINE_CYCLE_TIME_HR}h",
                    annotation_font_color="#9e9e9e",
                )

                fig_bar.update_layout(
                    template="plotly_white",
                    height=350,
                    margin=dict(l=40, r=20, t=20, b=40),
                    font=dict(family="Inter"),
                    yaxis_title="Cycle Time (hours)",
                    showlegend=False,
                    yaxis_range=[0, BASELINE_CYCLE_TIME_HR * 1.3],
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No comparison data available for this configuration.")
        else:
            st.info("Run CureOpt AI to see cycle time comparisons.")

    # ─── Recommendation Card ────────────────────────────────
    st.markdown('<div class="section-header">🎯 AI Recommendation</div>',
                unsafe_allow_html=True)

    opt_method = optimal.get("curing_method", curing_method_str)
    opt_time = optimal.get("cycle_time_hr", demould["demould_hour"])
    opt_strength = optimal.get("predicted_strength", demould["predicted_strength"])
    opt_reduction = optimal.get("time_reduction_pct", time_reduction)
    opt_cost_idx = optimal.get("cost_index", cost_index)
    opt_wc = optimal.get("water_cement_ratio", wc_ratio)
    opt_admixture = optimal.get("admixture_type", admixture_str)

    st.markdown(f"""
    <div class="rec-card">
        <h3>✅ Optimal Strategy — {opt_method}</h3>
        <div class="rec-metric">
            <div class="label">Curing Method</div>
            <div class="value">{opt_method}</div>
        </div>
        <div class="rec-metric">
            <div class="label">W/C Ratio</div>
            <div class="value">{opt_wc}</div>
        </div>
        <div class="rec-metric">
            <div class="label">Admixture</div>
            <div class="value">{opt_admixture}</div>
        </div>
        <div class="rec-metric">
            <div class="label">De-mould Time</div>
            <div class="value">{opt_time:.1f}h</div>
        </div>
        <div class="rec-metric">
            <div class="label">Strength</div>
            <div class="value">{opt_strength:.1f} MPa</div>
        </div>
        <div class="rec-metric">
            <div class="label">Cycle Time Δ</div>
            <div class="value" style="color: {'#1b5e20' if opt_reduction > 0 else '#b71c1c'}">{opt_reduction:+.1f}%</div>
        </div>
        <div class="rec-metric">
            <div class="label">Cost Index</div>
            <div class="value">{opt_cost_idx:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Scenario Comparison ────────────────────────────────
    if scenarios and len(scenarios) > 1:
        st.markdown('<div class="section-header">🔄 Top Scenarios Comparison</div>',
                    unsafe_allow_html=True)

        cols = st.columns(min(len(scenarios), 4))
        for idx, (col, sc) in enumerate(zip(cols, scenarios[:4])):
            with col:
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][idx]
                is_best = idx == 0

                st.markdown(f"""
                <div class="scenario-card" style="{'border: 2px solid #43a047;' if is_best else ''}">
                    <span class="scenario-rank">{idx + 1}</span>
                    <strong>{rank_emoji} {sc['curing_method']}</strong><br>
                    <small style="color: #5f6368;">w/c: {sc['water_cement_ratio']} · {sc['admixture_type']}</small>
                    <hr style="margin: 0.5rem 0; border-color: #e0e0e0;">
                    <div style="font-size: 0.85rem;">
                        ⏱️ <strong>{sc['cycle_time_hr']}h</strong> · 💪 {sc['predicted_strength']} MPa<br>
                        📉 <span style="color: {'#1b5e20' if sc['time_reduction_pct'] > 0 else '#b71c1c'}">{sc['time_reduction_pct']:+.1f}%</span>
                        · 💰 {sc['cost_index']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Radar chart for scenario comparison
        if len(scenarios) >= 2:
            st.markdown("")
            categories = ["Cycle Time", "Cost", "Strength", "Time Savings", "Efficiency"]

            fig_radar = go.Figure()
            radar_colors = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04"]

            for idx, sc in enumerate(scenarios[:4]):
                # Normalize values for radar (0-1 scale, higher is better)
                norm_time = max(0, 1 - sc["cycle_time_hr"] / 36)
                norm_cost = max(0, 1 - sc["cost_index"] / 2)
                norm_strength = min(1, sc["predicted_strength"] / 40)
                norm_savings = max(0, sc["time_reduction_pct"] / 50)
                norm_efficiency = max(0, (norm_time + norm_cost + norm_strength) / 3)

                values = [norm_time, norm_cost, norm_strength, norm_savings, norm_efficiency]
                values.append(values[0])  # close the polygon

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor=f"rgba({int(radar_colors[idx][1:3], 16)},{int(radar_colors[idx][3:5], 16)},{int(radar_colors[idx][5:7], 16)},0.1)",
                    line=dict(color=radar_colors[idx], width=2),
                    name=f"#{idx+1} {sc['curing_method']}",
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_white",
                height=400,
                margin=dict(l=60, r=60, t=40, b=40),
                font=dict(family="Inter"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ─── PDF Report Download ────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        try:
            from utils.pdf_report import generate_report

            input_params = {
                "cement_pct": cement_pct,
                "fly_ash_pct": fly_ash_pct,
                "water_cement_ratio": wc_ratio,
                "curing_method": curing_method_str,
                "admixture_type": admixture_str,
                "ambient_temp_C": ambient_temp,
                "humidity_pct": humidity,
                "required_mpa": required_mpa,
            }

            pdf_bytes = generate_report(
                recommendation=optimal,
                cost_breakdown=cost,
                demould_info=demould,
                input_params=input_params,
                region=region,
                scenarios=scenarios
            )

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"CureOpt_AI_Report_{region.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary",
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {e}")


# ─── Run Logic ──────────────────────────────────────────────────────────
if run_clicked:
    try:
        run_analysis()
    except FileNotFoundError:
        st.error(
            "⚠️ **Models not found!** Please run the training pipeline first:\n\n"
            "```bash\n"
            "python data/generate_dataset.py\n"
            "python models/train_model.py\n"
            "```"
        )
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
else:
    # Show welcome state
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🏗️</div>
        <h2 style="color: #1a237e; font-weight: 700;">Welcome to CureOpt AI</h2>
        <p style="color: #5f6368; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            Configure your batch parameters in the sidebar and click
            <strong>"🚀 Run CureOpt AI"</strong> to get AI-powered cycle time optimization recommendations.
        </p>
        <div class="kpi-row" style="justify-content: center; max-width: 800px; margin: 0 auto;">
            <div class="kpi-card">
                <div class="kpi-value">18–25%</div>
                <div class="kpi-label">Cycle Time Reduction</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">12–15%</div>
                <div class="kpi-label">Yard Space Freed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">8–10%</div>
                <div class="kpi-label">Cost Efficiency Gain</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">3× Year 1</div>
                <div class="kpi-label">Estimated ROI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pre-computed simulation results table
    st.markdown('<div class="section-header">📊 Pre-Computed Simulation Results</div>',
                unsafe_allow_html=True)

    sim_data = {
        "Location": ["Chennai", "Chennai", "Delhi", "Delhi", "Mumbai", "Mumbai"],
        "Curing Method": ["Normal", "Steam", "Normal", "Steam + Accelerator", "Normal", "Heated Chamber"],
        "Cycle Time (h)": [20, 13, 28, 16, 22, 14],
        "vs Baseline": ["-17%", "-46%", "+17%", "-33%", "-8%", "-42%"],
        "Cost Index": [0.93, 1.06, 0.91, 1.09, 0.94, 1.05],
        "MPa @ De-mould": [26.1, 28.4, 25.2, 27.1, 25.8, 27.6],
        "Recommended": ["", "✅ OPT", "", "✅ OPT", "", "✅ OPT"],
    }
    sim_df = pd.DataFrame(sim_data)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)
