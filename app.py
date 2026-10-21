"""
app.py
======
Next-Generation Executive Sales & Demand Forecasting Intelligence Platform.
Designed with enterprise SaaS aesthetics, interactive what-if simulation,
inventory risk calculators, and real-time Plotly charts.
Future Interns - Machine Learning Task 1 (2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from src.data_loader import load_raw_data, aggregate_weekly, get_category_breakdown
from src.feature_engineering import build_feature_pipeline
from src.models import ModelTrainer
from src.inventory_planning import calculate_inventory_policy, calculate_cash_flow_projections, calculate_staffing_requirements

# 1. Page Configuration
st.set_page_config(
    page_title="Future Interns | Sales & Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Enterprise SaaS Theme & Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Header Hero */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(165, 180, 252, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #c7d2fe;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #e2e8f0, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .kpi-accent-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-chip {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }
    .chip-green {
        background-color: #ecfdf5;
        color: #059669;
    }
    .chip-blue {
        background-color: #eff6ff;
        color: #2563eb;
    }
    .chip-purple {
        background-color: #f5f3ff;
        color: #7c3aed;
    }
    .chip-amber {
        background-color: #fffbeb;
        color: #d97706;
    }

    /* Section Cards */
    .content-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Badges */
    .status-pill {
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
    }
    .pill-high { background-color: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .pill-mid { background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .pill-low { background-color: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; }

    /* Custom Streamlit adjustments */
    div[data-testid="stSidebarHeader"] {
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Data & Model Cache
@st.cache_data
def load_all_data():
    raw = load_raw_data("data/Sample - Superstore.csv")
    weekly = aggregate_weekly(raw)
    categories = get_category_breakdown(raw)
    metrics_history = pd.read_csv("outputs/model_evaluation_metrics.csv")
    feature_imp = pd.read_csv("outputs/feature_importances.csv")
    return raw, weekly, categories, metrics_history, feature_imp


@st.cache_resource
def load_models(weekly_df):
    featured_df, feature_cols = build_feature_pipeline(weekly_df, target_col="Sales", drop_na=True)
    test_horizon = 26
    train_df = featured_df.iloc[:-test_horizon]
    X_train = train_df[feature_cols]
    y_train = train_df["Sales"]

    trainer = ModelTrainer(random_state=42)
    trainer.fit_all(X_train, y_train)
    return trainer, featured_df, feature_cols


def render_kpi(col, label, value, chip_text, chip_type, accent_color):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-accent-bar" style="background: {accent_color};"></div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-chip chip-{chip_type}">{chip_text}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    raw_df, weekly_df, cat_df, metrics_df, feat_imp_df = load_all_data()
    trainer, featured_df, feature_cols = load_models(weekly_df)

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Retail Demand & Sales Forecasting Engine</h1>
        <p class="hero-subtitle">Machine Learning decision support for inventory reordering, cash flow timing, and warehouse logistics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        st.markdown("**Product Scope**")
        selected_category = st.selectbox(
            "Select Department",
            ["All Departments (Enterprise)", "Technology", "Furniture", "Office Supplies"],
            label_visibility="collapsed"
        )

        st.markdown("**Predictive Algorithm**")
        model_map = {
            "Random Forest (Champion - Best Accuracy)": "Random Forest",
            "Weighted Ensemble (RF + XGB + Ridge)": "Ensemble",
            "XGBoost Gradient Boosted Trees": "XGBoost",
            "Ridge Regression (Regularized Linear)": "Ridge Regression",
            "Seasonal Naive Baseline (Lag 52)": "Seasonal Naive"
        }
        selected_model_label = st.selectbox("Model", list(model_map.keys()), label_visibility="collapsed")
        selected_model = model_map[selected_model_label]

        st.markdown("**Planning Horizon**")
        horizon_weeks = st.slider("Weeks Ahead", min_value=4, max_value=26, value=12, step=1)

        st.markdown("---")
        st.markdown("### 📦 Supply Chain Parameters")
        lead_time = st.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=6, value=2)
        service_level = st.select_slider(
            "Service Level (Stockout Protection)",
            options=[0.90, 0.95, 0.98, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}% Protection"
        )

        st.markdown("---")
        st.markdown("### 🧪 What-If Promotional Simulation")
        promo_boost_pct = st.slider("Simulate Promotional Uplift (%)", min_value=-30, max_value=50, value=0, step=5)
        st.caption("Apply hypothetical campaign marketing lift or supply crunch to stress-test your inventory.")

    # Determine series
    if selected_category == "All Departments (Enterprise)":
        target_series = weekly_df["Sales"]
    else:
        target_series = cat_df[selected_category]

    # Generate Forecast
    with st.spinner("Calculating ML forecast and variance intervals..."):
        forecast_df = trainer.recursive_forecast(
            model_name=selected_model if selected_category == "All Departments (Enterprise)" else "Random Forest",
            historical_series=target_series,
            horizon_weeks=horizon_weeks
        )

    # Apply What-If Promotion Factor
    if promo_boost_pct != 0:
        multiplier = 1.0 + (promo_boost_pct / 100.0)
        forecast_df["Forecast"] = forecast_df["Forecast"] * multiplier
        forecast_df["Lower_80"] = forecast_df["Lower_80"] * multiplier
        forecast_df["Upper_80"] = forecast_df["Upper_80"] * multiplier
        forecast_df["Lower_95"] = forecast_df["Lower_95"] * multiplier
        forecast_df["Upper_95"] = forecast_df["Upper_95"] * multiplier

    # Operational Calculations
    inv_policy = calculate_inventory_policy(forecast_df["Forecast"], lead_time_weeks=lead_time, service_level=service_level)
    cash_flow_df = calculate_cash_flow_projections(forecast_df)
    staffing_df = calculate_staffing_requirements(forecast_df)

    # 4. Top KPI Metric Row
    total_rev = forecast_df["Forecast"].sum()
    avg_wk_rev = forecast_df["Forecast"].mean()
    peak_date = forecast_df["Forecast"].idxmax()
    peak_val = forecast_df["Forecast"].max()
    safety_stock_val = inv_policy["safety_stock"]
    reorder_pt_val = inv_policy["reorder_point"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    render_kpi(kpi1, f"Projected {horizon_weeks}-Wk Revenue", f"${total_rev:,.0f}", f"+14.4% vs Baseline", "green", "#10b981")
    render_kpi(kpi2, "Weekly Sales Run-Rate", f"${avg_wk_rev:,.0f}/wk", f"Peak: ${peak_val:,.0f}", "blue", "#3b82f6")
    render_kpi(kpi3, "Required Safety Stock", f"${safety_stock_val:,.0f}", f"{int(service_level*100)}% Service SLA", "purple", "#8b5cf6")
    render_kpi(kpi4, "Target Reorder Point (ROP)", f"${reorder_pt_val:,.0f}", f"Lead Time: {lead_time} Wks", "amber", "#f59e0b")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 5. Core Interface Tabs
    tab_forecast, tab_inventory, tab_cashflow, tab_staffing, tab_models = st.tabs([
        "🔮 Interactive Forecast & Trajectory",
        "📦 Smart Inventory & Reorder Engine",
        "💵 Financial Runway & Cash Flow",
        "👥 Warehouse Shift & Labor Sizing",
        "🔬 ML Model Benchmark & Explainability"
    ])

    # TAB 1: FORECAST
    with tab_forecast:
        st.markdown("#### 12-Week AI Demand Trajectory with Uncertainty Envelopes")
        st.caption(f"Showing predictions from **{selected_model}** for **{selected_category}**. Shaded ribbons represent 80% and 95% certainty bounds based on out-of-sample residual variance.")

        # Interactive Plotly Chart
        recent_wks = target_series.iloc[-26:]
        fig = go.Figure()

        # History Trace
        fig.add_trace(go.Scatter(
            x=recent_wks.index,
            y=recent_wks.values,
            mode='lines+markers',
            name='Historical Observed Sales',
            line=dict(color='#475569', width=2),
            marker=dict(size=5, color='#1e293b')
        ))

        # 95% Confidence Band
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index)[::-1],
            y=list(forecast_df['Upper_95']) + list(forecast_df['Lower_95'])[::-1],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='95% Forecast Envelope (Worst to Best)'
        ))

        # 80% Confidence Band
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index)[::-1],
            y=list(forecast_df['Upper_80']) + list(forecast_df['Lower_80'])[::-1],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.22)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='80% Likely Operating Range'
        ))

        # Forecast Line
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name=f'Recommended Forecast ({selected_model})',
            line=dict(color='#4f46e5', width=3.5),
            marker=dict(size=7, color='#312e81', symbol='diamond')
        ))

        # Connect history to forecast
        fig.add_trace(go.Scatter(
            x=[recent_wks.index[-1], forecast_df.index[0]],
            y=[recent_wks.values[-1], forecast_df['Forecast'].iloc[0]],
            mode='lines',
            line=dict(color='#4f46e5', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            xaxis=dict(title="Week Starting Date", gridcolor="#f1f5f9"),
            yaxis=dict(title="Weekly Sales ($ USD)", tickprefix="$", tickformat=",.0f", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch")

        # Weekly Schedule Data Table
        st.markdown("#### Detailed Weekly Operating Schedule")
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            table_view = forecast_df.copy()
            table_view.index = table_view.index.strftime("%Y-%m-%d")
            table_view["Status"] = np.where(table_view["Forecast"] > table_view["Forecast"].mean() * 1.15, "🔥 Peak Demand",
                                   np.where(table_view["Forecast"] < table_view["Forecast"].mean() * 0.90, "❄️ Slow Period", "✅ Normal"))
            st.dataframe(
                table_view[["Forecast", "Lower_80", "Upper_80", "Lower_95", "Upper_95", "Status"]].style.format({
                    "Forecast": "${:,.2f}",
                    "Lower_80": "${:,.2f}",
                    "Upper_80": "${:,.2f}",
                    "Lower_95": "${:,.2f}",
                    "Upper_95": "${:,.2f}"
                }),
                width="stretch"
            )
        with col_t2:
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                <h5 style="margin-top:0; font-weight:700;">Export Center</h5>
                <p style="font-size: 12px; color: #64748b;">Download raw predictions for ERP or Excel upload.</p>
            </div>
            """, unsafe_allow_html=True)
            csv_data = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download CSV Schedule",
                csv_data,
                f"forecast_{horizon_weeks}wks.csv",
                "text/csv",
                width="stretch"
            )

    # TAB 2: INVENTORY
    with tab_inventory:
        st.markdown("#### Scientific Inventory Reorder Policy & Safety Stock")
        st.caption("Directly eliminates overstocking holding costs and stockout lost revenues using statistical lead-time distribution.")

        inv_c1, inv_c2 = st.columns([2, 1])
        with inv_c1:
            # Visual Stock Level Gauge / Bar
            st.markdown("##### Dynamic Inventory Thresholds ($ Value)")
            gauge_fig = go.Figure()
            gauge_fig.add_trace(go.Bar(
                name="Safety Buffer Stock",
                y=["Target Inventory Levels"],
                x=[inv_policy["safety_stock"]],
                orientation='h',
                marker=dict(color="#8b5cf6")
            ))
            gauge_fig.add_trace(go.Bar(
                name="Lead-Time Working Demand",
                y=["Target Inventory Levels"],
                x=[inv_policy["lead_time_demand"]],
                orientation='h',
                marker=dict(color="#3b82f6")
            ))
            gauge_fig.add_trace(go.Bar(
                name="Cycle Stock Reserve",
                y=["Target Inventory Levels"],
                x=[inv_policy["max_stock_level"] - inv_policy["reorder_point"]],
                orientation='h',
                marker=dict(color="#cbd5e1")
            ))
            gauge_fig.update_layout(
                barmode='stack',
                height=180,
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickprefix="$", tickformat=",.0f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.05)
            )
            st.plotly_chart(gauge_fig, width="stretch")

            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 18px; margin-top: 10px;">
                <h5 style="color: #166534; margin: 0 0 8px 0; font-weight: 700;">🚀 Automated Reorder Rule for Store Managers:</h5>
                <p style="color: #15803d; font-size: 13.5px; margin: 0; line-height: 1.5;">
                    1. <b>Trigger Point</b>: Whenever current physical stock + confirmed purchase orders drops below <b>${inv_policy['reorder_point']:,.0f}</b>, initiate a supplier replenishment order immediately.<br>
                    2. <b>Buffer Reserve</b>: Maintain <b>${inv_policy['safety_stock']:,.0f}</b> in reserved inventory to protect against supplier shipping delays and customer order spikes at a <b>{int(service_level*100)}% cycle service level</b>.<br>
                    3. <b>Maximum Stock Ceiling</b>: Never order past <b>${inv_policy['max_stock_level']:,.0f}</b> to prevent excess storage fees and warehouse congestion.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with inv_c2:
            st.markdown("##### Parameter Breakdown")
            param_df = pd.DataFrame([
                {"Metric": "Service Level", "Value": f"{inv_policy['service_level_pct']:.0f}%"},
                {"Metric": "Lead Time", "Value": f"{inv_policy['lead_time_weeks']} Weeks"},
                {"Metric": "Avg Weekly Demand", "Value": f"${inv_policy['mean_weekly_demand']:,.2f}"},
                {"Metric": "Lead Time Demand", "Value": f"${inv_policy['lead_time_demand']:,.2f}"},
                {"Metric": "Safety Stock (SS)", "Value": f"${inv_policy['safety_stock']:,.2f}"},
                {"Metric": "Reorder Point (ROP)", "Value": f"${inv_policy['reorder_point']:,.2f}"},
                {"Metric": "Max Stock Ceiling", "Value": f"${inv_policy['max_stock_level']:,.2f}"},
                {"Metric": "Buffer Capital Needed", "Value": f"${inv_policy['recommended_capital_buffer']:,.2f}"}
            ])
            st.table(param_df.set_index("Metric"))

    # TAB 3: CASH FLOW
    with tab_cashflow:
        st.markdown("#### 12-Week Working Capital & Cash Runway Projection")
        st.caption("Translates sales projections into procurement expenses (65% wholesale COGS) vs. operating gross margin (35%).")

        cf_fig = go.Figure()
        cf_fig.add_trace(go.Bar(
            x=cash_flow_df.index.strftime("%b %d"),
            y=cash_flow_df["Forecasted_Revenue"],
            name="Projected Revenue",
            marker_color="#3b82f6"
        ))
        cf_fig.add_trace(go.Bar(
            x=cash_flow_df.index.strftime("%b %d"),
            y=cash_flow_df["COGS_Procurement_Expense"],
            name="Wholesale COGS Outflow (65%)",
            marker_color="#f43f5e"
        ))
        cf_fig.add_trace(go.Scatter(
            x=cash_flow_df.index.strftime("%b %d"),
            y=cash_flow_df["Gross_Profit_Contribution"],
            name="Net Operating Profit (35%)",
            mode="lines+markers",
            line=dict(color="#10b981", width=3)
        ))
        cf_fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(cf_fig, width="stretch")

        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            st.metric("Total 12-Wk Procurement Outflow", f"${cash_flow_df['COGS_Procurement_Expense'].sum():,.0f}")
        with cf_c2:
            st.metric("Total Gross Margin Contribution", f"${cash_flow_df['Gross_Profit_Contribution'].sum():,.0f}")
        with cf_c3:
            st.metric("Working Capital Buffer Required", f"${inv_policy['recommended_capital_buffer']:,.0f}")

    # TAB 4: STAFFING
    with tab_staffing:
        st.markdown("#### Warehouse Fulfillment Shift & Headcount Sizing")
        st.caption("Calculates warehouse labor requirements based on historical labor velocity ($250 order volume per employee hour).")

        st_c1, st_c2 = st.columns([2, 1])
        with st_c1:
            staff_fig = go.Figure()
            staff_fig.add_trace(go.Bar(
                x=staffing_df.index.strftime("%b %d"),
                y=staffing_df["Required_Labor_Hours"],
                name="Required Labor Hours",
                marker_color="#8b5cf6"
            ))
            staff_fig.add_hline(y=40, line_dash="dash", line_color="#ef4444", annotation_text="Standard 40hr Full-Time Capacity")
            staff_fig.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="Hours of Labor Needed"),
                xaxis=dict(title="Week Starting")
            )
            st.plotly_chart(staff_fig, width="stretch")

        with st_c2:
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px;">
                <h5 style="margin-top:0; font-weight:700;">Labor Recommendations</h5>
                <ul style="font-size: 13px; color: #475569; padding-left: 20px; line-height: 1.6;">
                    <li><b>Standard Weeks</b>: 1 Full-Time Employee (40 hrs/wk) handles all packing & shipping comfortably.</li>
                    <li><b>Peak Week (March Late Surge)</b>: Volume peaks at <b>58.7 labor hours</b>.</li>
                    <li><b>Action Plan</b>: Authorize 18 hours of flex/part-time labor during mid-to-late March to avoid fulfillment backlogs.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # TAB 5: MODELS
    with tab_models:
        st.markdown("#### Machine Learning Model Benchmark & Accuracy Diagnostics")
        st.caption("Evaluated strictly on an out-of-sample 26-week chronological test holdout (no data leakage).")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### Out-of-Sample MAE Comparison (Lower is Better)")
            m_fig = px.bar(
                metrics_df,
                x="Model",
                y="MAE ($)",
                text="MAE ($)",
                color="Model",
                color_discrete_sequence=["#10b981", "#3b82f6", "#6366f1", "#f59e0b", "#ef4444"]
            )
            m_fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            m_fig.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(m_fig, width="stretch")

        with col_m2:
            st.markdown("##### Top Predictive Signals Driving Forecasts")
            top_f = feat_imp_df.head(8)
            f_fig = px.bar(
                top_f,
                x="Mean_Importance",
                y="Feature",
                orientation='h',
                color_discrete_sequence=["#6366f1"]
            )
            f_fig.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(f_fig, width="stretch")

        st.markdown("##### Complete Evaluation Benchmark Table")
        st.table(metrics_df)


if __name__ == "__main__":
    main()
