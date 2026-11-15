"""
Supply Chain Analytics - Inventory Decision Tools
Deterministic Python tools for core inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- Validates inputs strictly
- Returns structured results
- Returns intermediate calculations
- Returns the final answer
- Identifies underlying operations research assumptions
- Never invents missing numerical inputs
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data Models / Structured Results
# ---------------------------------------------------------------------------

@dataclass
class DeterministicModelResult:
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: float = 0.0
    verification_checks: List[Dict[str, str]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def get_eight_point_breakdown(self) -> Dict[str, Any]:
        """
        Returns structured dictionary containing the mandatory 8 items.
        """
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions
        }

    def format_eight_point_report(self) -> str:
        """
        Formats the 8 items into standard Markdown for explanation stages and reporting.
        """
        lines = []
        model_name = getattr(self, "model", "Inventory Model")
        lines.append(f"## {model_name} - Deterministic Calculation Report\n")

        # 1. Inputs Extracted
        lines.append("### 1. Inputs Extracted")
        for k, v in self.inputs_extracted.items():
            if k == "discount_schedule" and isinstance(v, list):
                lines.append(f"- **{k}**: {len(v)} tiers provided")
            elif isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.2f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # 2. Variable Definitions
        lines.append("### 2. Variable Definitions")
        for sym, defn in self.variable_definitions.items():
            lines.append(f"- **{sym}**: {defn}")
        lines.append("")

        # 3. Formula
        lines.append("### 3. Formula")
        lines.append(f"```text\n{self.formula}\n```\n")

        # 4. Formula with Substituted Values
        lines.append("### 4. Formula with Substituted Values")
        lines.append(f"```text\n{self.formula_with_substituted_values}\n```\n")

        # 5. Intermediate Calculations
        lines.append("### 5. Intermediate Calculations")
        for k, v in self.intermediate_calculations.items():
            if k == "tiers_evaluated" and isinstance(v, list):
                lines.append(f"- **{k}**:")
                for tier in v:
                    lines.append(f"  - Bracket {tier.get('bracket')}: unconstrained EOQ = {tier.get('unconstrained_eoq'):.2f}, feasible lot size = {tier.get('adjusted_feasible_lot_size'):.2f}, total annual cost = ${tier.get('total_cost'):,.2f}")
            elif isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`" if abs(v) < 1 else f"- **{k}**: `{v:,.2f}`")
            elif isinstance(v, (int, str)):
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # 6. Final Answer
        lines.append("### 6. Final Answer")
        lines.append(f"**Final Answer**: `{self.final_answer}`\n")

        # 7. Verification Checks
        lines.append("### 7. Verification Checks")
        for check in self.verification_checks:
            st = check.get("status", "PASS")
            emoji = "✅" if st == "PASS" else ("⚠️" if st == "WARN" else "❌")
            lines.append(f"- {emoji} **{check.get('check', 'Sanity Check')}** [{st}]: {check.get('message', check.get('detail', ''))}")
        lines.append("")

        # 8. Assumptions
        lines.append("### 8. Assumptions")
        for asm in self.assumptions:
            lines.append(f"- {asm}")
        lines.append("")

        return "\n".join(lines)


@dataclass
class EOQResult(DeterministicModelResult):
    model: str = "Economic Order Quantity (EOQ)"
    final_answer: float = 0.0
    eoq: float = 0.0
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_cost: float = 0.0
    unit_cost: float = 0.0
    orders_per_year: float = 0.0
    cycle_time_days: float = 0.0
    annual_order_cost: float = 0.0
    annual_holding_cost: float = 0.0
    annual_purchase_cost: float = 0.0
    total_policy_cost: float = 0.0
    total_annual_cost: float = 0.0
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    cost_sensitivity: Dict[str, float] = field(default_factory=dict)
    curve_data: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eight_point_breakdown"] = self.get_eight_point_breakdown()
        return d


@dataclass
class DiscountBracketResult:
    bracket_index: int
    min_qty: float
    max_qty: float
    unit_cost: float
    holding_cost: float
    raw_eoq: float
    feasible_qty: float
    is_breakpoint: bool
    annual_purchase_cost: float
    annual_order_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult(DeterministicModelResult):
    model: str = "Quantity Discount Analysis"
    final_answer: float = 0.0
    optimal_order_quantity: float = 0.0
    optimal_bracket_index: int = 0
    optimal_total_cost: float = 0.0
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    baseline_total_cost: float = 0.0
    annual_cost_savings: float = 0.0
    percentage_savings: float = 0.0
    brackets: List[DiscountBracketResult] = field(default_factory=list)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eight_point_breakdown"] = self.get_eight_point_breakdown()
        return d


@dataclass
class SafetyStockROPResult(DeterministicModelResult):
    model: str = "Safety Stock and Reorder Point (ROP)"
    final_answer: float = 0.0
    reorder_point: float = 0.0
    safety_stock: float = 0.0
    case_type: str = ""
    avg_demand_per_period: float = 0.0
    std_demand_per_period: float = 0.0
    lead_time_periods: float = 0.0
    std_lead_time_periods: float = 0.0
    service_level: float = 0.0
    z_score: float = 0.0
    lead_time_demand_mean: float = 0.0
    lead_time_demand_std: float = 0.0
    stockout_risk_per_cycle: float = 0.0
    annual_ss_holding_cost: Optional[float] = None
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    distribution_curve: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eight_point_breakdown"] = self.get_eight_point_breakdown()
        return d


@dataclass
class FixedPeriodReviewResult(DeterministicModelResult):
    model: str = "Fixed-Period Review (Periodic Order-Up-To)"
    final_answer: float = 0.0
    recommended_order_qty: float = 0.0
    order_up_to_level: float = 0.0
    target_inventory: float = 0.0
    protection_interval: float = 0.0
    protection_period: float = 0.0
    safety_stock: float = 0.0
    inventory_position: float = 0.0
    review_period_periods: float = 0.0
    lead_time_periods: float = 0.0
    avg_demand_per_period: float = 0.0
    std_demand_per_period: float = 0.0
    service_level: float = 0.0
    z_score: float = 0.0
    protection_demand_mean: float = 0.0
    protection_demand_std: float = 0.0
    current_on_hand: float = 0.0
    current_on_order: float = 0.0
    backorders: float = 0.0
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eight_point_breakdown"] = self.get_eight_point_breakdown()
        return d


@dataclass
class NewsvendorResult(DeterministicModelResult):
    model: str = "Newsvendor (Single-Period Probabilistic)"
    final_answer: float = 0.0
    optimal_order_quantity: float = 0.0
    underage_cost: float = 0.0
    overage_cost: float = 0.0
    critical_ratio: float = 0.0
    selling_price: float = 0.0
    unit_cost: float = 0.0
    salvage_value: float = 0.0
    stockout_penalty: float = 0.0
    mean_demand: float = 0.0
    std_demand: float = 0.0
    z_score: float = 0.0
    expected_lost_sales: float = 0.0
    expected_sales: float = 0.0
    expected_leftovers: float = 0.0
    expected_revenue: float = 0.0
    expected_purchasing_cost: float = 0.0
    expected_profit: float = 0.0
    max_possible_profit: float = 0.0
    stockout_probability: float = 0.0
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    distribution_curve: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eight_point_breakdown"] = self.get_eight_point_breakdown()
        return d


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ) Tool
# ---------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: float = 0.0,
    work_days_per_year: int = 250
) -> EOQResult:
    """
    Computes Economic Order Quantity (EOQ): Q = sqrt((2 * D * S) / H).
    
    Validates all inputs, never invents missing numerical inputs,
    returns intermediate calculations, final answer, and assumptions.
    """
    # 1. Input Validation (never invent missing numerical inputs)
    if annual_demand is None:
        raise ValueError("Missing required input: annual_demand (D) must be provided.")
    if order_cost is None:
        raise ValueError("Missing required input: order_cost (S) must be provided.")
    if holding_cost is None:
        raise ValueError("Missing required input: holding_cost (H) must be provided.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"Annual demand (D) must be a strictly positive number. Got: {annual_demand}")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"Order/setup cost (S) must be a strictly positive number. Got: {order_cost}")
    if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
        raise ValueError(f"Holding cost (H) must be a strictly positive number. Got: {holding_cost}")
    if not isinstance(unit_cost, (int, float)) or unit_cost < 0:
        raise ValueError(f"Unit cost cannot be negative. Got: {unit_cost}")
    if not isinstance(work_days_per_year, int) or work_days_per_year <= 0:
        raise ValueError(f"Working days per year must be a positive integer. Got: {work_days_per_year}")

    # 2. Intermediate & Core Calculations
    raw_eoq = math.sqrt((2.0 * annual_demand * order_cost) / holding_cost)
    raw_orders = annual_demand / raw_eoq
    raw_cycle_days = (raw_eoq / annual_demand) * work_days_per_year
    raw_order_cost = raw_orders * order_cost
    raw_holding_cost = (raw_eoq / 2.0) * holding_cost
    raw_purchase_cost = annual_demand * unit_cost
    raw_policy_cost = raw_order_cost + raw_holding_cost
    raw_total_cost = raw_purchase_cost + raw_policy_cost

    ann_order_cost = round(raw_order_cost, 2)
    ann_holding_cost = round(raw_holding_cost, 2)
    ann_purchase_cost = round(raw_purchase_cost, 2)
    total_policy_cost = round(ann_order_cost + ann_holding_cost, 2)
    total_annual_cost = round(ann_purchase_cost + total_policy_cost, 2)
    rounded_eoq = round(raw_eoq, 2)

    # Sensitivity analysis: deviance in Total Policy Cost for +/- 10% and +/- 25% Q
    sensitivity = {}
    for pct in [-25, -10, 10, 25]:
        q_perturbed = raw_eoq * (1.0 + pct / 100.0)
        c_perturbed = (annual_demand / q_perturbed) * order_cost + (q_perturbed / 2.0) * holding_cost
        pct_increase = ((c_perturbed - raw_policy_cost) / raw_policy_cost) * 100.0
        sensitivity[f"{pct:+d}% Q"] = round(pct_increase, 2)

    # Cost curve generation
    q_vals = np.linspace(max(1.0, 0.2 * raw_eoq), 2.5 * raw_eoq, 60).tolist()
    ordering_curve = [(annual_demand / q) * order_cost for q in q_vals]
    holding_curve = [(q / 2.0) * holding_cost for q in q_vals]
    total_curve = [oc + hc for oc, hc in zip(ordering_curve, holding_curve)]

    # 1. Inputs Extracted (deterministic capture)
    inputs_extracted = {
        "annual_demand": annual_demand,
        "order_cost": order_cost,
        "holding_cost": holding_cost,
        "unit_cost": unit_cost,
        "work_days_per_year": work_days_per_year
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual Demand in units per year",
        "S": "Fixed order or setup cost per replenishment order ($/order)",
        "H": "Holding or carrying cost per unit per year ($/unit/year)",
        "Q": "Economic Order Quantity (optimal lot size in units)",
        "C": "Unit purchase price ($/unit)",
        "N": "Number of orders placed per year = D / Q",
        "T": "Cycle time between orders in working days = (Q / D) * work_days_per_year",
        "AOC": "Annual Ordering Cost = (D / Q) * S",
        "AHC": "Annual Holding Cost = (Q / 2) * H",
        "TPC": "Total Annual Policy Cost = AOC + AHC",
        "TC": "Total Annual Cost = (D * C) + AOC + AHC"
    }

    # 3. Formula
    formula = "Q = sqrt((2 * D * S) / H)"

    # 4. Formula with Substituted Values
    num_product = 2.0 * annual_demand * order_cost
    ratio_val = num_product / holding_cost
    formula_with_substituted_values = (
        f"Q = sqrt((2 * {annual_demand:,.2f} * {order_cost:,.2f}) / {holding_cost:,.2f}) "
        f"= sqrt({num_product:,.2f} / {holding_cost:,.2f}) "
        f"= sqrt({ratio_val:,.2f}) = {raw_eoq:,.2f} units"
    )

    # 5. Intermediate Calculations
    intermediate = {
        "formula": "Q = sqrt(2 * D * S / H)",
        "annual_demand_D": annual_demand,
        "order_cost_S": order_cost,
        "holding_cost_H": holding_cost,
        "numerator_2DS": num_product,
        "ratio_2DS_over_H": ratio_val,
        "raw_eoq": raw_eoq,
        "orders_per_year_N": raw_orders,
        "cycle_time_days_T": raw_cycle_days,
        "annual_ordering_cost_AOC": raw_order_cost,
        "annual_holding_cost_AHC": raw_holding_cost,
        "annual_purchase_cost": raw_purchase_cost,
        "total_policy_cost": raw_policy_cost,
        "total_annual_cost": raw_total_cost,
        "cost_equilibrium_diff": abs(ann_order_cost - ann_holding_cost)
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "EOQ Positivity",
            "status": "PASS" if raw_eoq > 0 else "FAIL",
            "message": f"Calculated EOQ ({rounded_eoq:,.2f} units) is strictly greater than zero."
        },
        {
            "check": "Cost Parity Balance",
            "status": "PASS" if abs(ann_order_cost - ann_holding_cost) <= 1.0 else "WARN",
            "message": f"Annual Ordering Cost (${ann_order_cost:,.2f}) equals Annual Holding Cost (${ann_holding_cost:,.2f}) within rounding margin (${abs(ann_order_cost - ann_holding_cost):.2f})."
        },
        {
            "check": "Order Cadence Feasibility",
            "status": "PASS" if raw_orders > 0 else "FAIL",
            "message": f"Order frequency is {raw_orders:.2f} orders/year (every {raw_cycle_days:.1f} working days)."
        }
    ]

    # 8. Assumptions
    assumptions = [
        "Annual demand (D) is constant, uniform, and known with certainty over the planning year.",
        "Order / setup cost (S) is fixed per replenishment, independent of lot size.",
        "Holding cost (H) is constant per unit per year and linear with average inventory.",
        "Replenishment lead time is zero (instantaneous batch delivery).",
        "Shortages, stockouts, and backorders are not permitted.",
        "Single item with constant purchase price (no quantity discounts)."
    ]

    return EOQResult(
        model="Economic Order Quantity (EOQ)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=rounded_eoq,
        verification_checks=verification_checks,
        assumptions=assumptions,
        eoq=rounded_eoq,
        annual_demand=round(annual_demand, 2),
        order_cost=round(order_cost, 2),
        holding_cost=round(holding_cost, 2),
        unit_cost=round(unit_cost, 2),
        orders_per_year=round(raw_orders, 2),
        cycle_time_days=round(raw_cycle_days, 1),
        annual_order_cost=ann_order_cost,
        annual_holding_cost=ann_holding_cost,
        annual_purchase_cost=ann_purchase_cost,
        total_policy_cost=total_policy_cost,
        total_annual_cost=total_annual_cost,
        cost_sensitivity=sensitivity,
        curve_data={
            "quantities": [round(q, 1) for q in q_vals],
            "ordering_costs": [round(c, 2) for c in ordering_curve],
            "holding_costs": [round(c, 2) for c in holding_curve],
            "total_costs": [round(c, 2) for c in total_curve]
        }
    )


# ---------------------------------------------------------------------------
# 2. Quantity Discount Tool
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    holding_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None
) -> QuantityDiscountResult:
    """
    Evaluates all-units quantity discount brackets.
    Compares total annual costs (purchase + ordering + holding) across feasible quantities.
    Supports both fixed holding cost ($/unit/year) and percentage holding cost (% of price).
    """
    # 1. Validation
    if annual_demand is None:
        raise ValueError("Missing required input: annual_demand (D) must be provided.")
    if order_cost is None:
        raise ValueError("Missing required input: order_cost (S) must be provided.")
    if not discount_schedule:
        raise ValueError("Missing required input: discount_schedule must be a non-empty list of tiers.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError("Annual demand must be a strictly positive number.")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError("Order cost must be a strictly positive number.")

    if holding_rate is None and fixed_holding_cost is None:
        # Check if all brackets provide holding_cost
        for i, b in enumerate(discount_schedule):
            if "holding_cost" not in b or b["holding_cost"] is None:
                raise ValueError("Must provide either global holding_rate, fixed_holding_cost, or per-bracket holding_cost.")

    brackets_evaluated: List[DiscountBracketResult] = []
    intermediate_tiers: List[Dict[str, Any]] = []

    for i, bracket in enumerate(discount_schedule):
        if not isinstance(bracket, dict):
            raise TypeError(f"Bracket {i} must be a dictionary.")
        if "unit_cost" not in bracket or bracket["unit_cost"] is None:
            raise ValueError(f"Bracket {i} is missing unit_cost.")

        unit_c = float(bracket["unit_cost"])
        if unit_c <= 0:
            raise ValueError(f"Bracket {i} unit_cost must be positive.")

        min_q = float(bracket.get("min_qty", 1.0))
        max_q = float(bracket.get("max_qty", math.inf))
        if min_q < 0 or max_q < min_q:
            raise ValueError(f"Bracket {i} has invalid quantity bounds: [{min_q}, {max_q}].")

        # Resolve holding cost for this bracket
        if "holding_cost" in bracket and bracket["holding_cost"] is not None:
            h_cost = float(bracket["holding_cost"])
        elif fixed_holding_cost is not None and fixed_holding_cost > 0:
            h_cost = float(fixed_holding_cost)
        elif holding_rate is not None and holding_rate > 0:
            h_cost = float(holding_rate * unit_c)
        else:
            raise ValueError(f"Unable to determine holding cost for bracket {i}.")

        if h_cost <= 0:
            raise ValueError(f"Holding cost for bracket {i} must be strictly positive.")

        # Compute raw EOQ
        raw_eoq = math.sqrt((2.0 * annual_demand * order_cost) / h_cost)

        # Check bracket feasibility & breakpoint upward adjustment
        is_breakpoint = False
        if raw_eoq < min_q:
            feasible_q = min_q
            is_breakpoint = True
        elif raw_eoq > max_q:
            feasible_q = max_q
        else:
            feasible_q = raw_eoq

        ann_purchase = annual_demand * unit_c
        ann_order = (annual_demand / feasible_q) * order_cost
        ann_holding = (feasible_q / 2.0) * h_cost
        total_cost = ann_purchase + ann_order + ann_holding

        bracket_res = DiscountBracketResult(
            bracket_index=i,
            min_qty=min_q,
            max_qty=max_q,
            unit_cost=round(unit_c, 3),
            holding_cost=round(h_cost, 3),
            raw_eoq=round(raw_eoq, 2),
            feasible_qty=round(feasible_q, 2),
            is_breakpoint=is_breakpoint,
            annual_purchase_cost=round(ann_purchase, 2),
            annual_order_cost=round(ann_order, 2),
            annual_holding_cost=round(ann_holding, 2),
            total_annual_cost=round(total_cost, 2)
        )
        brackets_evaluated.append(bracket_res)

        intermediate_tiers.append({
            "bracket": i + 1,
            "range": f"[{min_q}, {max_q}]",
            "unit_price": unit_c,
            "unit_holding_cost": h_cost,
            "unconstrained_eoq": raw_eoq,
            "adjusted_feasible_lot_size": feasible_q,
            "is_breakpoint_adjustment": is_breakpoint,
            "purchase_cost": ann_purchase,
            "ordering_cost": ann_order,
            "holding_cost": ann_holding,
            "total_cost": total_cost
        })

    # Optimal bracket is the one with minimum total annual cost
    optimal_bracket = min(brackets_evaluated, key=lambda b: b.total_annual_cost)
    baseline_bracket = brackets_evaluated[0]
    savings = baseline_bracket.total_annual_cost - optimal_bracket.total_annual_cost
    pct_savings = (savings / baseline_bracket.total_annual_cost) * 100.0 if baseline_bracket.total_annual_cost > 0 else 0.0

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": annual_demand,
        "order_cost": order_cost,
        "holding_rate": holding_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "discount_schedule": discount_schedule
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual Demand in units per year",
        "S": "Fixed order / setup cost per replenishment order ($/order)",
        "i": "Price bracket / discount tier index (0-indexed)",
        "min_qty": "Minimum lot size threshold required to qualify for discount tier",
        "max_qty": "Upper quantity bound for discount tier",
        "P_i": "Unit purchase price in discount tier i ($/unit)",
        "H_i": "Holding cost per unit per year in discount tier i ($/unit/year)",
        "Q_raw_i": "Unconstrained Wilson EOQ for tier i = sqrt((2 * D * S) / H_i)",
        "Q_feas_i": "Feasible lot size for tier i = max(min_qty, min(Q_raw_i, max_qty))",
        "Purchase_Cost": "Annual purchase outlay = D * P_i",
        "Order_Cost": "Annual ordering expenditure = (D / Q_feas_i) * S",
        "Holding_Cost": "Annual carrying cost = (Q_feas_i / 2) * H_i",
        "TC_i": "Total Annual Cost = Purchase_Cost + Order_Cost + Holding_Cost"
    }

    # 3. Formula
    formula = (
        "Q_raw_i = sqrt((2 * D * S) / H_i); "
        "Q_feas_i = max(min_qty_i, min(Q_raw_i, max_qty_i)); "
        "TC_i = (D * P_i) + (D / Q_feas_i * S) + (Q_feas_i / 2 * H_i)"
    )

    # 4. Formula with Substituted Values (detail per tier and optimal choice)
    sub_lines = []
    for b in brackets_evaluated:
        max_repr = f"{b.max_qty:,.0f}" if b.max_qty != math.inf else "inf"
        sub_lines.append(
            f"Tier {b.bracket_index + 1} [{b.min_qty:,.0f}, {max_repr}] @ ${b.unit_cost:.2f}: "
            f"Q_raw = sqrt((2 * {annual_demand:,.0f} * {order_cost:.2f}) / {b.holding_cost:.2f}) = {b.raw_eoq:.2f} -> Q_feas = {b.feasible_qty:.2f}; "
            f"TC = ({annual_demand:,.0f} * {b.unit_cost:.2f}) + ({annual_demand:,.0f} / {b.feasible_qty:.2f} * {order_cost:.2f}) + ({b.feasible_qty:.2f} / 2 * {b.holding_cost:.2f}) "
            f"= {b.annual_purchase_cost:.2f} + {b.annual_order_cost:.2f} + {b.annual_holding_cost:.2f} = ${b.total_annual_cost:,.2f}"
        )
    formula_with_substituted_values = "\n".join(sub_lines)

    # 5. Intermediate Calculations
    intermediate = {
        "tiers_evaluated": intermediate_tiers,
        "baseline_tier_cost": baseline_bracket.total_annual_cost,
        "optimal_tier_index": optimal_bracket.bracket_index,
        "optimal_order_quantity": optimal_bracket.feasible_qty,
        "optimal_total_cost": optimal_bracket.total_annual_cost,
        "net_annual_savings": max(0.0, savings),
        "percentage_savings": max(0.0, pct_savings)
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Bracket Feasibility",
            "status": "PASS" if optimal_bracket.min_qty <= optimal_bracket.feasible_qty <= optimal_bracket.max_qty else "FAIL",
            "message": f"Optimal lot size {optimal_bracket.feasible_qty:,.0f} falls within Tier {optimal_bracket.bracket_index + 1} qualification range."
        },
        {
            "check": "Global Cost Minimization",
            "status": "PASS" if all(optimal_bracket.total_annual_cost <= b.total_annual_cost for b in brackets_evaluated) else "FAIL",
            "message": f"Optimal Tier {optimal_bracket.bracket_index + 1} achieves minimum annual total cost (${optimal_bracket.total_annual_cost:,.2f}) across all feasible brackets."
        },
        {
            "check": "Non-Negative Savings",
            "status": "PASS" if savings >= 0 else "FAIL",
            "message": f"Annual discount savings (${max(0.0, savings):,.2f}, {max(0.0, pct_savings):.1f}%) is non-negative relative to baseline."
        }
    ]

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount: the discounted unit price applies to all units ordered in the batch.",
        "Annual demand is known with certainty and constant over time.",
        "Holding costs reflect either a constant percentage of unit cost or a fixed dollar cost per unit.",
        "Supplier guarantees delivery of ordered lot sizes without volume rationing.",
        "Sufficient capital and warehouse capacity exist to store the larger discount batch sizes."
    ]

    return QuantityDiscountResult(
        model="Quantity Discount Analysis",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=optimal_bracket.feasible_qty,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_order_quantity=optimal_bracket.feasible_qty,
        optimal_bracket_index=optimal_bracket.bracket_index,
        optimal_total_cost=optimal_bracket.total_annual_cost,
        annual_demand=round(annual_demand, 2),
        order_cost=round(order_cost, 2),
        holding_rate=holding_rate,
        fixed_holding_cost=fixed_holding_cost,
        baseline_total_cost=baseline_bracket.total_annual_cost,
        annual_cost_savings=round(max(0.0, savings), 2),
        percentage_savings=round(max(0.0, pct_savings), 2),
        brackets=brackets_evaluated
    )


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point (ROP) Tool
# ---------------------------------------------------------------------------

def calculate_rop(
    avg_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    service_level: float = 0.95,
    unit_holding_cost: Optional[float] = None
) -> SafetyStockROPResult:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP).
    Supports all 4 operations research cases:
      1. Constant demand and constant lead time
      2. Variable demand and constant lead time
      3. Constant demand and variable lead time
      4. Variable demand and variable lead time:
         sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
         SS = z * sigma_LT
         ROP = d * L + SS
    """
    # 1. Validation
    if avg_demand is None:
        raise ValueError("Missing required input: avg_demand (d) must be provided.")
    if lead_time is None:
        raise ValueError("Missing required input: lead_time (L) must be provided.")

    if not isinstance(avg_demand, (int, float)) or avg_demand <= 0:
        raise ValueError("Average demand per period must be a strictly positive number.")
    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError("Lead time must be a strictly positive number.")
    if not isinstance(std_demand, (int, float)) or std_demand < 0:
        raise ValueError("Demand standard deviation cannot be negative.")
    if not isinstance(std_lead_time, (int, float)) or std_lead_time < 0:
        raise ValueError("Lead-time standard deviation cannot be negative.")
    if not (0.50 <= service_level < 1.0):
        raise ValueError("Service level must be between 0.50 (50%) and 0.9999 (99.99%).")

    # 2. Case Determination & Sigma_LT Formula
    ltd_mean = avg_demand * lead_time

    if std_demand > 0 and std_lead_time > 0:
        # Case 4: Variable demand and variable lead time
        case_type = "Variable demand and variable lead time"
        ltd_var = lead_time * (std_demand ** 2) + (avg_demand ** 2) * (std_lead_time ** 2)
        sigma_lt = math.sqrt(ltd_var)
    elif std_demand > 0 and std_lead_time == 0:
        # Case 2: Variable demand and constant lead time
        case_type = "Variable demand and constant lead time"
        ltd_var = lead_time * (std_demand ** 2)
        sigma_lt = std_demand * math.sqrt(lead_time)
    elif std_demand == 0 and std_lead_time > 0:
        # Case 3: Constant demand and variable lead time
        case_type = "Constant demand and variable lead time"
        ltd_var = (avg_demand ** 2) * (std_lead_time ** 2)
        sigma_lt = avg_demand * std_lead_time
    else:
        # Case 1: Constant demand and constant lead time
        case_type = "Constant demand and constant lead time"
        ltd_var = 0.0
        sigma_lt = 0.0

    z_score = float(stats.norm.ppf(service_level)) if sigma_lt > 0 else 0.0
    safety_stock = z_score * sigma_lt
    reorder_point = ltd_mean + safety_stock
    stockout_risk = 1.0 - service_level

    annual_ss_cost = None
    if unit_holding_cost and unit_holding_cost > 0:
        annual_ss_cost = round(safety_stock * unit_holding_cost, 2)

    # Lead time demand distribution curve data
    curve = {}
    if sigma_lt > 0:
        x_min = max(0.0, ltd_mean - 3.5 * sigma_lt)
        x_max = ltd_mean + 3.5 * sigma_lt
        x_vals = np.linspace(x_min, x_max, 100).tolist()
        pdf_vals = stats.norm.pdf(x_vals, loc=ltd_mean, scale=sigma_lt).tolist()
        curve = {
            "demand_levels": [round(x, 1) for x in x_vals],
            "densities": [round(float(p), 6) for p in pdf_vals]
        }

    # 1. Inputs Extracted
    inputs_extracted = {
        "avg_demand": avg_demand,
        "lead_time": lead_time,
        "std_demand": std_demand,
        "std_lead_time": std_lead_time,
        "service_level": service_level,
        "unit_holding_cost": unit_holding_cost
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average customer demand per period (units/period)",
        "L": "Average replenishment lead time (periods)",
        "sigma_d": "Standard deviation of demand per period",
        "sigma_L": "Standard deviation of lead time across replenishment cycles",
        "sigma_LT": "Combined standard deviation of lead time demand",
        "CSL": "Cycle Service Level (target non-stockout probability per cycle)",
        "z": "Standard normal critical factor = Phi^(-1)(CSL)",
        "SS": "Safety Stock buffer in units = z * sigma_LT",
        "ROP": "Reorder Point in units = (d * L) + SS"
    }

    # 3. Formula & 4. Formula with Substituted Values
    if std_demand > 0 and std_lead_time > 0:
        formula = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2); SS = z * sigma_LT; ROP = (d * L) + SS"
        term1 = lead_time * (std_demand ** 2)
        term2 = (avg_demand ** 2) * (std_lead_time ** 2)
        formula_with_substituted_values = (
            f"sigma_LT = sqrt({lead_time} * ({std_demand})^2 + ({avg_demand})^2 * ({std_lead_time})^2) "
            f"= sqrt({term1:.2f} + {term2:.2f}) = sqrt({ltd_var:.2f}) = {sigma_lt:.2f}; "
            f"SS = {z_score:.3f} * {sigma_lt:.2f} = {safety_stock:.2f}; "
            f"ROP = ({avg_demand} * {lead_time}) + {safety_stock:.2f} = {ltd_mean:.2f} + {safety_stock:.2f} = {reorder_point:.2f} units"
        )
    elif std_demand > 0 and std_lead_time == 0:
        formula = "sigma_LT = sigma_d * sqrt(L); SS = z * sigma_LT; ROP = (d * L) + SS"
        formula_with_substituted_values = (
            f"sigma_LT = {std_demand} * sqrt({lead_time}) = {sigma_lt:.2f}; "
            f"SS = {z_score:.3f} * {sigma_lt:.2f} = {safety_stock:.2f}; "
            f"ROP = ({avg_demand} * {lead_time}) + {safety_stock:.2f} = {ltd_mean:.2f} + {safety_stock:.2f} = {reorder_point:.2f} units"
        )
    elif std_demand == 0 and std_lead_time > 0:
        formula = "sigma_LT = d * sigma_L; SS = z * sigma_LT; ROP = (d * L) + SS"
        formula_with_substituted_values = (
            f"sigma_LT = {avg_demand} * {std_lead_time} = {sigma_lt:.2f}; "
            f"SS = {z_score:.3f} * {sigma_lt:.2f} = {safety_stock:.2f}; "
            f"ROP = ({avg_demand} * {lead_time}) + {safety_stock:.2f} = {ltd_mean:.2f} + {safety_stock:.2f} = {reorder_point:.2f} units"
        )
    else:
        formula = "sigma_LT = 0; SS = 0; ROP = d * L"
        formula_with_substituted_values = (
            f"sigma_LT = 0.0; SS = 0.0; ROP = {avg_demand} * {lead_time} = {reorder_point:.2f} units"
        )

    # 5. Intermediate Calculations
    intermediate = {
        "case_type": case_type,
        "lead_time_demand_mean_dL": ltd_mean,
        "lead_time_demand_variance": ltd_var,
        "sigma_LT": sigma_lt,
        "z_score": z_score,
        "safety_stock_SS": safety_stock,
        "reorder_point_ROP": reorder_point,
        "cycle_service_level": service_level,
        "cycle_stockout_risk": stockout_risk
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "ROP Non-Deficit",
            "status": "PASS" if reorder_point >= ltd_mean else "FAIL",
            "message": f"ROP ({reorder_point:,.2f} units) covers expected lead time demand ({ltd_mean:,.2f} units)."
        },
        {
            "check": "Safety Stock Buffer",
            "status": "PASS" if safety_stock >= 0 else "FAIL",
            "message": f"Safety stock ({safety_stock:,.2f} units) is non-negative for target service level {service_level*100:.1f}%."
        },
        {
            "check": "Service Level Boundary",
            "status": "PASS" if 0.50 <= service_level < 1.0 else "FAIL",
            "message": f"Cycle service level ({service_level*100:.1f}%) is within valid probability bounds [50%, 99.99%)."
        }
    ]

    # 8. Assumptions
    assumptions = [
        f"Inventory model condition: {case_type}.",
        "Demand across successive periods is independent and identically distributed.",
        "Lead times across orders are independent and identically distributed (when stochastic).",
        "Lead time demand follows a Normal distribution for buffer determination.",
        "Continuous review inventory policy: order placed immediately when inventory position drops to ROP.",
        "Stockouts are backordered until the replenishment delivery arrives."
    ]

    return SafetyStockROPResult(
        model="Safety Stock and Reorder Point (ROP)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=round(reorder_point, 2),
        verification_checks=verification_checks,
        assumptions=assumptions,
        reorder_point=round(reorder_point, 2),
        safety_stock=round(safety_stock, 2),
        case_type=case_type,
        avg_demand_per_period=round(avg_demand, 2),
        std_demand_per_period=round(std_demand, 2),
        lead_time_periods=round(lead_time, 2),
        std_lead_time_periods=round(std_lead_time, 2),
        service_level=round(service_level, 4),
        z_score=round(z_score, 3),
        lead_time_demand_mean=round(ltd_mean, 2),
        lead_time_demand_std=round(sigma_lt, 2),
        stockout_risk_per_cycle=round(stockout_risk, 4),
        annual_ss_holding_cost=annual_ss_cost,
        distribution_curve=curve
    )


# Alias for backward compatibility
calculate_safety_stock_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review Tool (Periodic Order-Up-To Level)
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    avg_demand: Optional[float] = None,
    std_demand: float = 0.0,
    service_level: float = 0.95,
    current_inventory: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None
) -> FixedPeriodReviewResult:
    """
    Computes Periodic Review (Order-Up-To Level / (R, S) policy):
    Protection period = review period + lead time (R + L)
    Target inventory = demand during protection period + safety stock
    Order quantity = max(0, target inventory - inventory position)
    """
    # 1. Validation
    if review_period is None:
        raise ValueError("Missing required input: review_period (R) must be provided.")
    if lead_time is None:
        raise ValueError("Missing required input: lead_time (L) must be provided.")
    if avg_demand is None:
        raise ValueError("Missing required input: avg_demand (d) must be provided.")

    if not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError("Review period must be a strictly positive number.")
    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError("Lead time must be a strictly positive number.")
    if not isinstance(avg_demand, (int, float)) or avg_demand <= 0:
        raise ValueError("Average demand must be a strictly positive number.")
    if not isinstance(std_demand, (int, float)) or std_demand < 0:
        raise ValueError("Demand standard deviation cannot be negative.")
    if not (0.50 <= service_level < 1.0):
        raise ValueError("Service level must be between 0.50 and 0.9999.")

    # 2. Protection Period & Target Stock (S)
    protection_period = review_period + lead_time
    protection_mean = avg_demand * protection_period
    protection_std = std_demand * math.sqrt(protection_period) if std_demand > 0 else 0.0

    if safety_stock is not None and safety_stock >= 0:
        ss_val = float(safety_stock)
        z_score = (ss_val / protection_std) if protection_std > 0 else 0.0
    else:
        z_score = float(stats.norm.ppf(service_level)) if protection_std > 0 else 0.0
        ss_val = z_score * protection_std

    target_inventory = protection_mean + ss_val

    # 3. Inventory Position & Replenishment Order Quantity
    if inventory_position is not None:
        inventory_pos = float(inventory_position)
    else:
        inventory_pos = current_inventory + on_order - backorders

    order_qty = max(0.0, target_inventory - inventory_pos)

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period": review_period,
        "lead_time": lead_time,
        "avg_demand": avg_demand,
        "std_demand": std_demand,
        "service_level": service_level,
        "current_inventory": current_inventory,
        "on_order": on_order,
        "backorders": backorders,
        "safety_stock": safety_stock,
        "inventory_position": inventory_position
    }

    # 2. Variable Definitions
    variable_definitions = {
        "R": "Review period duration in periods",
        "L": "Replenishment lead time in periods",
        "P": "Protection period duration = R + L",
        "d": "Average demand rate per period (units/period)",
        "sigma_d": "Demand standard deviation per period",
        "mu_P": "Expected demand over protection interval = d * (R + L)",
        "sigma_P": "Demand standard deviation over protection interval = sigma_d * sqrt(R + L)",
        "z": "Normal critical value for target cycle service level",
        "SS": "Safety Stock buffer = z * sigma_P",
        "S": "Target inventory / Order-up-to level = mu_P + SS",
        "IP": "Effective inventory position = current_inventory + on_order - backorders",
        "Q": "Recommended replenishment order quantity = max(0, S - IP)"
    }

    # 3. Formula
    formula = (
        "P = R + L; "
        "mu_P = d * P; "
        "sigma_P = sigma_d * sqrt(P); "
        "S = mu_P + z * sigma_P; "
        "IP = On_Hand + On_Order - Backorders; "
        "Q = max(0, S - IP)"
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = (
        f"P = {review_period} + {lead_time} = {protection_period:.1f} periods; "
        f"mu_P = {avg_demand} * {protection_period:.1f} = {protection_mean:.2f} units; "
        f"sigma_P = {std_demand} * sqrt({protection_period:.1f}) = {protection_std:.2f}; "
        f"SS = {z_score:.3f} * {protection_std:.2f} = {ss_val:.2f} units; "
        f"S = {protection_mean:.2f} + {ss_val:.2f} = {target_inventory:.2f} units; "
        f"IP = {current_inventory} + {on_order} - {backorders} = {inventory_pos:.2f} units; "
        f"Q = max(0, {target_inventory:.2f} - {inventory_pos:.2f}) = {order_qty:.2f} units"
    )

    # 5. Intermediate Calculations
    intermediate = {
        "review_period_R": review_period,
        "lead_time_L": lead_time,
        "protection_period": protection_period,
        "protection_demand_mean": protection_mean,
        "protection_demand_std": protection_std,
        "z_score": z_score,
        "safety_stock": ss_val,
        "target_inventory_S": target_inventory,
        "on_hand": current_inventory,
        "on_order": on_order,
        "backorders": backorders,
        "effective_inventory_position_IP": inventory_pos,
        "recommended_order_quantity_Q": order_qty
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Protection Window Positivity",
            "status": "PASS" if protection_period > 0 else "FAIL",
            "message": f"Protection period ({protection_period:.1f} periods = R + L) is strictly positive."
        },
        {
            "check": "Target Inventory Sufficiency",
            "status": "PASS" if target_inventory >= protection_mean else "FAIL",
            "message": f"Target inventory level S ({target_inventory:,.2f} units) safely exceeds expected demand ({protection_mean:,.2f} units)."
        },
        {
            "check": "Order Quantity Non-Negativity",
            "status": "PASS" if order_qty >= 0 else "FAIL",
            "message": f"Recommended order quantity ({order_qty:,.2f} units) is non-negative max(0, S - IP)."
        }
    ]

    # 8. Assumptions
    assumptions = [
        "Inventory levels are reviewed periodically at fixed intervals of length R.",
        "Protection interval covers both review cycle and replenishment lead time (R + L).",
        "Target inventory (S) must satisfy customer demand until the next order arrives.",
        "Demand across periods is independent and normally distributed.",
        "Stockouts during the protection window are backlogged until arrival.",
        "Pipeline orders (on-order) arrive reliably within lead time L."
    ]

    return FixedPeriodReviewResult(
        model="Fixed-Period Review (Periodic Order-Up-To)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=round(order_qty, 2),
        verification_checks=verification_checks,
        assumptions=assumptions,
        recommended_order_qty=round(order_qty, 2),
        order_up_to_level=round(target_inventory, 2),
        target_inventory=round(target_inventory, 2),
        protection_interval=round(protection_period, 2),
        protection_period=round(protection_period, 2),
        safety_stock=round(ss_val, 2),
        inventory_position=round(inventory_pos, 2),
        review_period_periods=round(review_period, 2),
        lead_time_periods=round(lead_time, 2),
        avg_demand_per_period=round(avg_demand, 2),
        std_demand_per_period=round(std_demand, 2),
        service_level=round(service_level, 4),
        z_score=round(z_score, 3),
        protection_demand_mean=round(protection_mean, 2),
        protection_demand_std=round(protection_std, 2),
        current_on_hand=round(current_inventory, 2),
        current_on_order=round(on_order, 2),
        backorders=round(backorders, 2)
    )


# Alias for backward compatibility
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. Newsvendor Tool (Single-Period Probabilistic Demand)
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    salvage_value: float = 0.0,
    stockout_penalty: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None
) -> NewsvendorResult:
    """
    Computes optimal single-period Newsvendor order quantity:
    - Underage cost (Cu): profit margin lost per unit unmet
    - Overage cost (Co): loss per unit unsold
    - Critical ratio: CR = Cu / (Cu + Co)
    - Recommended order quantity: Q* = mean + z * std
    """
    # 1. Validation
    if mean_demand is None:
        raise ValueError("Missing required input: mean_demand (μ) must be provided.")
    if std_demand is None:
        raise ValueError("Missing required input: std_demand (σ) must be provided.")

    if not isinstance(mean_demand, (int, float)) or mean_demand <= 0:
        raise ValueError("Mean demand must be a strictly positive number.")
    if not isinstance(std_demand, (int, float)) or std_demand <= 0:
        raise ValueError("Standard deviation of demand must be a strictly positive number.")

    # 2. Determine Underage Cost (Cu) and Overage Cost (Co)
    if underage_cost is not None and overage_cost is not None:
        if underage_cost <= 0 or overage_cost <= 0:
            raise ValueError("Underage and overage costs must be strictly positive.")
        cu = float(underage_cost)
        co = float(overage_cost)
        sp = float(selling_price) if selling_price is not None else (cu + (unit_cost or co))
        uc = float(unit_cost) if unit_cost is not None else co
    elif selling_price is not None and unit_cost is not None:
        if selling_price <= 0 or unit_cost <= 0:
            raise ValueError("Selling price and unit cost must be strictly positive.")
        if unit_cost >= selling_price:
            raise ValueError("Unit cost must be strictly less than selling price for a viable commercial product.")
        if salvage_value > unit_cost:
            raise ValueError("Salvage value cannot exceed unit cost.")
        cu = selling_price - unit_cost + stockout_penalty
        co = unit_cost - salvage_value
        sp = float(selling_price)
        uc = float(unit_cost)
    else:
        raise ValueError("Must provide either (selling_price, unit_cost) or (underage_cost, overage_cost).")

    # 3. Critical Ratio & Optimal Quantity Q*
    cr = cu / (cu + co)
    z_score = float(stats.norm.ppf(cr))
    q_opt = max(0.0, mean_demand + z_score * std_demand)

    # 4. Standard normal unit loss function: L(z) = phi(z) - z * (1 - Phi(z))
    phi_z = stats.norm.pdf(z_score)
    Phi_z = stats.norm.cdf(z_score)
    unit_loss = phi_z - z_score * (1.0 - Phi_z)

    expected_lost_sales = std_demand * unit_loss
    expected_sales = mean_demand - expected_lost_sales
    expected_leftovers = max(0.0, q_opt - expected_sales)

    expected_revenue = (sp * expected_sales) + (salvage_value * expected_leftovers)
    expected_purchase_cost = uc * q_opt
    expected_penalty_cost = stockout_penalty * expected_lost_sales
    expected_profit = expected_revenue - expected_purchase_cost - expected_penalty_cost

    max_possible_profit = (sp - uc) * mean_demand
    stockout_prob = 1.0 - cr

    # Distribution curve for visualization
    x_min = max(0.0, mean_demand - 3.5 * std_demand)
    x_max = mean_demand + 3.5 * std_demand
    x_vals = np.linspace(x_min, x_max, 100).tolist()
    pdf_vals = stats.norm.pdf(x_vals, loc=mean_demand, scale=std_demand).tolist()

    # 1. Inputs Extracted
    inputs_extracted = {
        "selling_price": selling_price,
        "unit_cost": unit_cost,
        "mean_demand": mean_demand,
        "std_demand": std_demand,
        "salvage_value": salvage_value,
        "stockout_penalty": stockout_penalty,
        "underage_cost": underage_cost,
        "overage_cost": overage_cost
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Retail selling price per unit ($/unit)",
        "C": "Purchase / acquisition cost per unit ($/unit)",
        "V": "Salvage value recovered per leftover unsold unit ($/unit)",
        "s": "Stockout goodwill penalty per unmet customer demand ($/unit)",
        "Cu": "Marginal underage cost of stocking out = P - C + s",
        "Co": "Marginal overage cost of excess unsold inventory = C - V",
        "CR": "Critical ratio (critical fractile) = Cu / (Cu + Co)",
        "mu": "Mean seasonal / single-period demand in units",
        "sigma": "Standard deviation of seasonal demand in units",
        "z_star": "Standard normal critical factor = Phi^(-1)(CR)",
        "Q_star": "Optimal stocking order quantity = max(0, mu + z_star * sigma)"
    }

    # 3. Formula
    formula = (
        "Cu = P - C + s; "
        "Co = C - V; "
        "CR = Cu / (Cu + Co); "
        "z* = Phi^(-1)(CR); "
        "Q* = max(0, mu + z* * sigma)"
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = (
        f"Cu = {sp:.2f} - {uc:.2f} + {stockout_penalty:.2f} = {cu:.2f}; "
        f"Co = {uc:.2f} - {salvage_value:.2f} = {co:.2f}; "
        f"CR = {cu:.2f} / ({cu:.2f} + {co:.2f}) = {cu:.2f} / {cu + co:.2f} = {cr:.4f}; "
        f"z* = Phi^(-1)({cr:.4f}) = {z_score:.3f}; "
        f"Q* = max(0, {mean_demand:.2f} + ({z_score:.3f} * {std_demand:.2f})) = "
        f"max(0, {mean_demand:.2f} + {z_score * std_demand:.2f}) = {q_opt:.2f} units"
    )

    # 5. Intermediate Calculations
    intermediate = {
        "underage_cost_Cu": cu,
        "overage_cost_Co": co,
        "critical_ratio_CR": cr,
        "z_score": z_score,
        "optimal_order_quantity_Q_star": q_opt,
        "expected_sales": expected_sales,
        "expected_lost_sales": expected_lost_sales,
        "expected_leftovers": expected_leftovers,
        "expected_revenue": expected_revenue,
        "expected_purchase_cost": expected_purchase_cost,
        "expected_profit": expected_profit,
        "perfect_information_profit": max_possible_profit,
        "stockout_probability": stockout_prob
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Marginal Cost Positivity",
            "status": "PASS" if cu > 0 and co > 0 else "FAIL",
            "message": f"Underage cost (${cu:,.2f}) and overage cost (${co:,.2f}) are strictly positive."
        },
        {
            "check": "Critical Ratio Probability Bounds",
            "status": "PASS" if 0 < cr < 1 else "FAIL",
            "message": f"Critical fractile ({cr:.4f}) falls strictly between 0 and 1."
        },
        {
            "check": "Order Quantity Non-Negativity",
            "status": "PASS" if q_opt >= 0 else "FAIL",
            "message": f"Optimal order quantity ({q_opt:,.2f} units) is non-negative."
        }
    ]

    # 8. Assumptions
    assumptions = [
        "Single-period planning horizon: one single ordering decision before demand is realized.",
        "Product is perishable, seasonal, or subject to obsolescence at period end.",
        "Demand is normally distributed with known mean (μ) and standard deviation (σ).",
        "Leftover units are liquidated at salvage value V without holding over to future seasons.",
        "Unsatisfied customer demand is completely lost (underage cost = lost profit contribution)."
    ]

    return NewsvendorResult(
        model="Newsvendor (Single-Period Probabilistic)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=round(q_opt, 2),
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_order_quantity=round(q_opt, 2),
        underage_cost=round(cu, 2),
        overage_cost=round(co, 2),
        critical_ratio=round(cr, 4),
        selling_price=round(sp, 2),
        unit_cost=round(uc, 2),
        salvage_value=round(salvage_value, 2),
        stockout_penalty=round(stockout_penalty, 2),
        mean_demand=round(mean_demand, 2),
        std_demand=round(std_demand, 2),
        z_score=round(z_score, 3),
        expected_lost_sales=round(expected_lost_sales, 2),
        expected_sales=round(expected_sales, 2),
        expected_leftovers=round(expected_leftovers, 2),
        expected_revenue=round(expected_revenue, 2),
        expected_purchasing_cost=round(expected_purchase_cost, 2),
        expected_profit=round(expected_profit, 2),
        max_possible_profit=round(max_possible_profit, 2),
        stockout_probability=round(stockout_prob, 4),
        distribution_curve={
            "demand_levels": [round(x, 1) for x in x_vals],
            "densities": [round(float(p), 6) for p in pdf_vals]
        }
    )
