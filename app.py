"""
Supply Chain Analytics: Inventory Decision Agent
Streamlit Web Application
Implements the 7-stage architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from inventory_agent import InventoryDecisionAgent
from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
    calculate_newsvendor
)

# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics | Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Agent
if "agent" not in st.session_state:
    st.session_state.agent = InventoryDecisionAgent()

# ---------------------------------------------------------------------------
# Sidebar & Sample Prompts
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/delivery-box.png", width=64)
    st.title("Inventory Decision Agent")
    st.caption("AI-Powered Supply Chain Analytics Engine")
    
    st.markdown("---")
    st.subheader("Decision Architecture")
    st.markdown(
        """
        1. 💬 **Customer Question**
        2. 🧭 **Identify Model**
        3. 🔍 **Extract Inputs**
        4. ⚙️ **Select Python Tool**
        5. 🧮 **Calculate**
        6. 🛡️ **Verify**
        7. 📢 **Explain**
        """
    )
    st.markdown("---")
    st.subheader("Supported Inventory Models")
    st.markdown(
        """
        - **1. Economic Order Quantity (EOQ)**
        - **2. Quantity Discount**
        - **3. Safety Stock & Reorder Point (ROP)**
        - **4. Fixed-Period Review (Periodic OUTL)**
        - **5. Newsvendor (Perishable Single-Period)**
        """
    )

# ---------------------------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------------------------
st.title("📦 Supply Chain Analytics Inventory Decision Agent")
st.markdown(
    "Ask natural-language inventory questions across classical operations research and supply chain models. "
    "The agent identifies the optimal model, extracts key parameters, runs pure Python calculations, verifies mathematical consistency, and produces managerial recommendations."
)

tab_nlp, tab_direct = st.tabs(["💬 Natural Language Decision Agent", "🧮 Direct Model Calculators"])

# ===========================================================================
# TAB 1: Natural Language Agent Interface
# ===========================================================================
with tab_nlp:
    st.markdown("### Step 1: Customer Question")
    
    presets = {
        "Select a sample scenario...": "",
        "Scenario 1: EOQ Batch Sizing": "We sell 12,000 units a year. Each order costs $60 to place and holding cost is $2.50 per unit per year. What is our optimal order quantity?",
        "Scenario 2: All-Units Quantity Discount": "Our annual demand is 5,000 units and ordering cost is $50. The holding rate is 20%. Tier 1: 1-499 units @ $10.00; Tier 2: 500-999 units @ $9.20; Tier 3: 1000+ units @ $8.50. What order quantity minimizes total cost?",
        "Scenario 3: Safety Stock & Continuous ROP": "Our average daily demand is 80 units with a standard deviation of 12 units. Supplier lead time is 5 days. We target a 95% cycle service level. What should our reorder point and safety stock be?",
        "Scenario 4: Fixed-Period Review (Periodic OUTL)": "We review store inventory every 14 days. Supplier delivery lead time is 7 days. Average daily demand is 30 units with a std dev of 6 units. Target service level is 95%. Current inventory on-hand is 180 units and on-order is 50 units. How much should we order today?",
        "Scenario 5: Seasonal Newsvendor": "For our holiday ski jacket, retail selling price is $120, wholesale unit cost is $50, and salvage clearance price is $20. Forecasted seasonal demand is normally distributed with mean 600 units and std dev 120 units. What is the optimal order quantity?"
    }

    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        chosen_preset = st.selectbox("Load Sample Inventory Scenario:", list(presets.keys()))
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Reset Input"):
            st.session_state["nl_query"] = ""

    default_text = presets[chosen_preset] if chosen_preset != "Select a sample scenario..." else st.session_state.get("nl_query", "")

    user_query = st.text_area(
        "Enter your inventory or supply chain question:",
        value=default_text,
        height=100,
        placeholder="e.g. Annual demand is 10,000 units, ordering cost is $40, holding cost is $2 per unit. What is our EOQ?"
    )

    run_pipeline = st.button("🚀 Analyze & Solve Decision", type="primary", use_container_width=True)

    if run_pipeline and user_query.strip():
        agent = st.session_state.agent
        with st.spinner("Executing 7-stage inventory decision pipeline..."):
            res = agent.process_question(user_query.strip())

        st.markdown("---")

        # Visual Pipeline Tracker
        st.markdown("### Pipeline Execution Trace")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("1. Model Identified", res.model_identified)
        c2.metric("2. Confidence", f"{res.identification_confidence*100:.0f}%")
        c3.metric("3. Tool Selected", res.tool_selected)
        c4.metric("4. Verification", "PASSED ✅" if res.verification_passed else "FAILED ⚠️")
        c5.metric("5. Decisions Generated", f"{len(res.managerial_recommendations)} Actions")

        # Stage 1 & 2 Details
        with st.expander("🔍 Stage 1 & 2: Identification & Input Extraction Details", expanded=True):
            col_id, col_inputs = st.columns(2)
            with col_id:
                st.markdown(f"**Identified Model:** `{res.model_name_display}`")
                st.markdown(f"**Confidence:** `{res.identification_confidence:.2f}`")
                st.info(f"**Reasoning:** {res.identification_reasoning}")
            with col_inputs:
                st.markdown("**Normalized Parameter Inputs:**")
                norm_df = pd.DataFrame(
                    [{"Parameter": k, "Value": str(v)} for k, v in res.normalized_inputs.items() if k != "discount_schedule"]
                )
                st.dataframe(norm_df, hide_index=True, use_container_width=True)

        # Stage 3 & 4: Numerical Results & Visualization
        if res.missing_inputs:
            st.warning(f"⚠️ **Action Required**: The following required parameters are missing to solve this problem: **{', '.join(res.missing_inputs)}**")
            st.markdown(res.explanation_details)
        else:
            st.markdown("### 🧮 Stage 4: Numerical Solution & KPIs")
            calc = res.calculation_result

            if res.model_identified == agent.MODEL_EOQ:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Lot Size (EOQ)", f"{calc.get('eoq', 0):,.0f} units")
                k2.metric("Orders per Year", f"{calc.get('orders_per_year', 0):.1f} orders")
                k3.metric("Cycle Time", f"{calc.get('cycle_time_days', 0):.1f} days")
                k4.metric("Total Policy Cost", f"${calc.get('total_policy_cost', 0):,.2f}")

                # Plot EOQ Cost Curve
                curve = calc.get("curve_data", {})
                if curve:
                    chart_df = pd.DataFrame({
                        "Order Quantity (Q)": curve["quantities"] * 3,
                        "Cost ($)": curve["ordering_costs"] + curve["holding_costs"] + curve["total_costs"],
                        "Cost Type": (["Annual Ordering Cost"] * len(curve["quantities"]) +
                                     ["Annual Holding Cost"] * len(curve["quantities"]) +
                                     ["Total Policy Cost"] * len(curve["quantities"]))
                    })
                    eoq_val = calc.get("eoq", 0)
                    base = alt.Chart(chart_df).encode(x="Order Quantity (Q):Q")
                    lines = base.mark_line(size=2.5).encode(
                        y=alt.Y("Cost ($):Q", title="Annual Cost ($)"),
                        color=alt.Color("Cost Type:N", scale=alt.Scale(range=["#e74c3c", "#3498db", "#2ecc71"]))
                    )
                    rule = alt.Chart(pd.DataFrame({"eoq": [eoq_val]})).mark_rule(
                        color="#2c3e50", strokeDash=[4, 4], size=2
                    ).encode(x="eoq:Q")
                    
                    eoq_chart = (lines + rule).properties(
                        title=f"EOQ Cost Trade-Off Curve (Optimal Wilson Lot Size Q* = {eoq_val:,.0f} units)",
                        height=350
                    )
                    st.altair_chart(eoq_chart, use_container_width=True)

            elif res.model_identified == agent.MODEL_QUANTITY_DISCOUNT:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Order Quantity", f"{calc.get('optimal_order_quantity', 0):,.0f} units")
                k2.metric("Optimal Bracket", f"Tier {calc.get('optimal_bracket_index', 0) + 1}")
                k3.metric("Annual Savings", f"${calc.get('annual_cost_savings', 0):,.2f}")
                k4.metric("Savings %", f"{calc.get('percentage_savings', 0):.1f}%")

                brackets = calc.get("brackets", [])
                if brackets:
                    b_df = pd.DataFrame(brackets)
                    b_df["Tier"] = [f"Tier {b['bracket_index'] + 1} (${b['unit_cost']:.2f})" for b in brackets]
                    st.dataframe(
                        b_df[["Tier", "min_qty", "feasible_qty", "annual_purchase_cost", "annual_order_cost", "annual_holding_cost", "total_annual_cost"]],
                        column_config={
                            "Tier": "Price Bracket",
                            "min_qty": "Min Order",
                            "feasible_qty": "Feasible Qty",
                            "annual_purchase_cost": st.column_config.NumberColumn("Purchase Cost ($)", format="$%.2f"),
                            "annual_order_cost": st.column_config.NumberColumn("Order Cost ($)", format="$%.2f"),
                            "annual_holding_cost": st.column_config.NumberColumn("Holding Cost ($)", format="$%.2f"),
                            "total_annual_cost": st.column_config.NumberColumn("Total Cost ($)", format="$%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    bar_chart = alt.Chart(b_df).mark_bar().encode(
                        x=alt.X("Tier:N", title="Discount Tier"),
                        y=alt.Y("total_annual_cost:Q", title="Total Annual Cost ($)"),
                        color=alt.condition(
                            alt.datum.bracket_index == calc.get("optimal_bracket_index", 0),
                            alt.value("#27ae60"),
                            alt.value("#95a5a6")
                        )
                    ).properties(title="Total Cost Comparison Across Price Brackets", height=300)
                    st.altair_chart(bar_chart, use_container_width=True)

            elif res.model_identified == agent.MODEL_SAFETY_STOCK_ROP:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Safety Stock (SS)", f"{calc.get('safety_stock', 0):,.0f} units")
                k2.metric("Reorder Point (ROP)", f"{calc.get('reorder_point', 0):,.0f} units")
                k3.metric("Service Level (CSL)", f"{calc.get('service_level', 0)*100:.1f}%")
                k4.metric("Lead Time Demand Mean", f"{calc.get('lead_time_demand_mean', 0):,.0f} units")

                curve = calc.get("distribution_curve", {})
                if curve:
                    c_df = pd.DataFrame({
                        "Demand during Lead Time": curve["demand_levels"],
                        "Probability Density": curve["densities"]
                    })
                    rop_val = calc.get("reorder_point", 0)
                    mean_val = calc.get("lead_time_demand_mean", 0)

                    base = alt.Chart(c_df).encode(x="Demand during Lead Time:Q")
                    density_line = base.mark_area(opacity=0.3, color="#3498db").encode(y="Probability Density:Q")
                    
                    mean_rule = alt.Chart(pd.DataFrame({"x": [mean_val]})).mark_rule(color="#7f8c8d", strokeDash=[3, 3]).encode(x="x:Q")
                    rop_rule = alt.Chart(pd.DataFrame({"x": [rop_val]})).mark_rule(color="#e74c3c", size=2.5).encode(x="x:Q")

                    chart = (density_line + mean_rule + rop_rule).properties(
                        title=f"Lead Time Demand Distribution (Mean: {mean_val:.0f}, ROP Buffer: {rop_val:.0f})",
                        height=320
                    )
                    st.altair_chart(chart, use_container_width=True)

            elif res.model_identified == agent.MODEL_FIXED_PERIOD_REVIEW:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Order-Up-To Level (S)", f"{calc.get('order_up_to_level', 0):,.0f} units")
                k2.metric("Recommended Order Qty", f"{calc.get('recommended_order_qty', 0):,.0f} units")
                k3.metric("Protection Interval (R+L)", f"{calc.get('protection_interval', 0):.0f} periods")
                k4.metric("Inventory Position", f"{calc.get('inventory_position', 0):,.0f} units")

                # Protection breakdown bar chart
                breakdown_df = pd.DataFrame([
                    {"Category": "Expected Protection Demand", "Units": calc.get("protection_demand_mean", 0)},
                    {"Category": "Safety Stock Buffer", "Units": calc.get("safety_stock", 0)},
                    {"Category": "Current Inventory Position", "Units": calc.get("inventory_position", 0)},
                    {"Category": "Recommended Order (Q)", "Units": calc.get("recommended_order_qty", 0)}
                ])
                bar = alt.Chart(breakdown_df).mark_bar().encode(
                    x=alt.X("Category:N", sort=None),
                    y=alt.Y("Units:Q"),
                    color=alt.Color("Category:N", legend=None)
                ).properties(title="Periodic Review Inventory Allocation Breakdown", height=300)
                st.altair_chart(bar, use_container_width=True)

            elif res.model_identified == agent.MODEL_NEWSVENDOR:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Order Qty (Q*)", f"{calc.get('optimal_order_quantity', 0):,.0f} units")
                k2.metric("Critical Fractile (CR)", f"{calc.get('critical_ratio', 0):.4f}")
                k3.metric("Expected Profit", f"${calc.get('expected_profit', 0):,.2f}")
                k4.metric("Expected Lost Sales", f"{calc.get('expected_lost_sales', 0):.1f} units")

                curve = calc.get("distribution_curve", {})
                if curve:
                    c_df = pd.DataFrame({
                        "Demand": curve["demand_levels"],
                        "Density": curve["densities"]
                    })
                    q_opt = calc.get("optimal_order_quantity", 0)
                    mean_dem = calc.get("mean_demand", 0)
                    base = alt.Chart(c_df).encode(x="Demand:Q")
                    density_line = base.mark_area(opacity=0.35, color="#9b59b6").encode(y="Density:Q")
                    q_rule = alt.Chart(pd.DataFrame({"q": [q_opt]})).mark_rule(color="#27ae60", size=2.5).encode(x="q:Q")
                    m_rule = alt.Chart(pd.DataFrame({"m": [mean_dem]})).mark_rule(color="#7f8c8d", strokeDash=[4, 4]).encode(x="m:Q")

                    chart = (density_line + m_rule + q_rule).properties(
                        title=f"Newsvendor Demand Curve & Critical Fractile Order Level Q* = {q_opt:.0f} (Mean = {mean_dem:.0f})",
                        height=320
                    )
                    st.altair_chart(chart, use_container_width=True)

        # Stage 5: Verification Checks
        st.markdown("### 🛡️ Stage 5: Model & Mathematical Verification")
        v_cols = st.columns(len(res.verification_checks))
        for i, check in enumerate(res.verification_checks):
            with v_cols[i % len(v_cols)]:
                if check.status == "PASS":
                    st.success(f"**{check.check_name}**\n\n{check.message}")
                elif check.status == "WARN":
                    st.warning(f"**{check.check_name}**\n\n{check.message}")
                else:
                    st.error(f"**{check.check_name}**\n\n{check.message}")

        # Stage 6 & 7: Explain & Recommendations
        st.markdown("### 📢 Stage 6 & 7: Decision Explanation & Managerial Guidance")
        st.info(f"### Executive Summary\n{res.explanation_summary}")
        
        with st.expander("📊 Detailed Economic Trade-Offs & Operations Analysis", expanded=True):
            st.markdown(res.explanation_details)

        st.markdown("#### Actionable ERP & Supply Chain Directives:")
        for r in res.managerial_recommendations:
            st.markdown(f"- ✅ {r}")

# ===========================================================================
# TAB 2: Direct Model Calculators (Structured Forms)
# ===========================================================================
with tab_direct:
    st.markdown("### Direct Operations Research Calculators")
    st.caption("Calculate exact inventory metrics directly without natural language queries.")

    calc_model = st.selectbox(
        "Select Inventory Model:",
        [
            "1. Economic Order Quantity (EOQ)",
            "2. Quantity Discount Analysis",
            "3. Safety Stock & Reorder Point (ROP)",
            "4. Fixed-Period Review (Periodic Order-Up-To)",
            "5. Newsvendor (Perishable Single-Period)"
        ]
    )

    if "1. Economic Order Quantity" in calc_model:
        st.subheader("Economic Order Quantity (EOQ) Calculator")
        c1, c2, c3, c4 = st.columns(4)
        d_val = c1.number_input("Annual Demand (D)", min_value=1.0, value=10000.0, step=100.0)
        s_val = c2.number_input("Order Cost (S)", min_value=0.1, value=50.0, step=5.0)
        h_val = c3.number_input("Holding Cost (H $/unit/yr)", min_value=0.01, value=2.0, step=0.1)
        c_val = c4.number_input("Unit Cost (Optional)", min_value=0.0, value=10.0, step=1.0)

        if st.button("Compute EOQ", type="primary"):
            res = calculate_eoq(d_val, s_val, h_val, c_val)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("EOQ (Q*)", f"{res.eoq:,.0f} units")
            r2.metric("Orders / Year", f"{res.orders_per_year:.2f}")
            r3.metric("Cycle Time", f"{res.cycle_time_days:.1f} days")
            r4.metric("Policy Cost", f"${res.total_policy_cost:,.2f}")
            with st.expander("📋 Complete 8-Point Calculation Breakdown (Deterministic Python)", expanded=True):
                st.markdown(res.format_eight_point_report())
            st.json(res.to_dict())

    elif "2. Quantity Discount" in calc_model:
        st.subheader("Quantity Discount Optimizer")
        c1, c2, c3 = st.columns(3)
        d_val = c1.number_input("Annual Demand", min_value=1.0, value=5000.0, step=100.0)
        s_val = c2.number_input("Order Cost ($)", min_value=0.1, value=50.0, step=5.0)
        hr_val = c3.number_input("Holding Rate (%)", min_value=1.0, max_value=100.0, value=20.0) / 100.0

        st.markdown("**Discount Tiers:**")
        t1, t2, t3 = st.columns(3)
        p1 = t1.number_input("Tier 1 (1 - 499) Price ($)", value=10.0)
        p2 = t2.number_input("Tier 2 (500 - 999) Price ($)", value=9.20)
        p3 = t3.number_input("Tier 3 (1000+) Price ($)", value=8.50)

        schedule = [
            {"min_qty": 1, "max_qty": 499, "unit_cost": p1},
            {"min_qty": 500, "max_qty": 999, "unit_cost": p2},
            {"min_qty": 1000, "max_qty": math.inf, "unit_cost": p3}
        ]

        if st.button("Compute Optimal Discount Lot Size", type="primary"):
            res = calculate_quantity_discount(d_val, s_val, schedule, hr_val)
            r1, r2, r3 = st.columns(3)
            r1.metric("Optimal Lot Size", f"{res.optimal_order_quantity:,.0f} units")
            r2.metric("Optimal Tier", f"Tier {res.optimal_bracket_index + 1}")
            r3.metric("Annual Savings", f"${res.annual_cost_savings:,.2f}")
            with st.expander("📋 Complete 8-Point Calculation Breakdown (Deterministic Python)", expanded=True):
                st.markdown(res.format_eight_point_report())
            st.json(res.to_dict())

    elif "3. Safety Stock & Reorder Point" in calc_model:
        st.subheader("Safety Stock & Reorder Point (ROP) Calculator")
        c1, c2, c3 = st.columns(3)
        d_val = c1.number_input("Avg Daily Demand (d)", min_value=0.1, value=50.0, step=5.0)
        sig_d = c2.number_input("Std Dev Daily Demand (σ_d)", min_value=0.0, value=8.0, step=1.0)
        l_val = c3.number_input("Lead Time in Days (L)", min_value=0.1, value=6.0, step=1.0)
        c4, c5 = st.columns(2)
        sig_l = c4.number_input("Std Dev Lead Time (σ_L)", min_value=0.0, value=0.0, step=0.5)
        csl_val = c5.slider("Target Service Level (CSL)", min_value=0.50, max_value=0.999, value=0.95, step=0.01)

        if st.button("Compute Safety Stock & ROP", type="primary"):
            res = calculate_safety_stock_rop(d_val, l_val, sig_d, csl_val, sig_l)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Safety Stock", f"{res.safety_stock:,.1f} units")
            r2.metric("Reorder Point", f"{res.reorder_point:,.1f} units")
            r3.metric("z-score", f"{res.z_score:.2f}")
            r4.metric("Stockout Risk", f"{res.stockout_risk_per_cycle*100:.1f}%")
            with st.expander("📋 Complete 8-Point Calculation Breakdown (Deterministic Python)", expanded=True):
                st.markdown(res.format_eight_point_report())
            st.json(res.to_dict())

    elif "4. Fixed-Period Review" in calc_model:
        st.subheader("Fixed-Period Review (Periodic OUTL) Calculator")
        c1, c2, c3 = st.columns(3)
        r_val = c1.number_input("Review Period in Days (R)", min_value=1.0, value=14.0, step=1.0)
        l_val = c2.number_input("Lead Time in Days (L)", min_value=1.0, value=7.0, step=1.0)
        d_val = c3.number_input("Average Daily Demand (d)", min_value=0.1, value=25.0, step=1.0)
        c4, c5, c6 = st.columns(3)
        sig_d = c4.number_input("Demand Std Dev (σ_d)", min_value=0.0, value=5.0, step=1.0)
        csl = c5.slider("Cycle Service Level", min_value=0.50, max_value=0.999, value=0.95, step=0.01)
        on_hand = c6.number_input("Current On-Hand Inventory", min_value=0.0, value=150.0, step=10.0)

        if st.button("Compute Order-Up-To Level", type="primary"):
            res = calculate_fixed_period_review(r_val, l_val, d_val, sig_d, csl, on_hand)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Target Stock (S)", f"{res.order_up_to_level:,.0f} units")
            r2.metric("Order Today (Q)", f"{res.recommended_order_qty:,.0f} units")
            r3.metric("Safety Stock", f"{res.safety_stock:,.1f} units")
            r4.metric("Protection Interval", f"{res.protection_interval:.0f} days")
            with st.expander("📋 Complete 8-Point Calculation Breakdown (Deterministic Python)", expanded=True):
                st.markdown(res.format_eight_point_report())
            st.json(res.to_dict())

    elif "5. Newsvendor" in calc_model:
        st.subheader("Newsvendor Model Calculator")
        c1, c2, c3 = st.columns(3)
        p_val = c1.number_input("Retail Price (P)", min_value=0.1, value=100.0, step=5.0)
        c_val = c2.number_input("Wholesale Unit Cost (C)", min_value=0.1, value=40.0, step=5.0)
        v_val = c3.number_input("Salvage Value (V)", min_value=0.0, value=15.0, step=5.0)
        c4, c5 = st.columns(2)
        mu_val = c4.number_input("Expected Demand Mean (μ)", min_value=1.0, value=1000.0, step=50.0)
        sig_val = c5.number_input("Demand Std Dev (σ)", min_value=1.0, value=200.0, step=10.0)

        if st.button("Compute Optimal Newsvendor Quantity", type="primary"):
            res = calculate_newsvendor(p_val, c_val, mu_val, sig_val, v_val)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Optimal Lot Size (Q*)", f"{res.optimal_order_quantity:,.0f} units")
            r2.metric("Critical Fractile (CR)", f"{res.critical_ratio:.4f}")
            r3.metric("Expected Profit", f"${res.expected_profit:,.2f}")
            r4.metric("Expected Leftovers", f"{res.expected_leftovers:,.1f} units")
            with st.expander("📋 Complete 8-Point Calculation Breakdown (Deterministic Python)", expanded=True):
                st.markdown(res.format_eight_point_report())
            st.json(res.to_dict())
