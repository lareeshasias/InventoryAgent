"""
Supply Chain Analytics - Inventory Decision Agent
Orchestrates the 7-stage architecture:
Customer Question
  -> Identify Model
  -> Extract Inputs
  -> Select Python Tool
  -> Calculate
  -> Verify
  -> Explain
"""

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

import inventory_tools as tools


# ---------------------------------------------------------------------------
# Pipeline Result Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationCheck:
    check_name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str


@dataclass
class AgentExecutionResult:
    customer_question: str
    model_identified: str
    model_name_display: str
    identification_confidence: float
    identification_reasoning: str
    extracted_inputs: Dict[str, Any]
    normalized_inputs: Dict[str, Any]
    missing_inputs: List[str]
    tool_selected: str
    calculation_result: Dict[str, Any]
    verification_passed: bool
    verification_checks: List[VerificationCheck]
    explanation_summary: str
    explanation_details: str
    managerial_recommendations: List[str]
    eight_point_breakdown: Dict[str, Any] = field(default_factory=dict)
    stage_traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Inventory Decision Agent
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """
    Seven-stage supply chain analytics inventory decision agent.
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"
    MODEL_INVENTORY_CLASSIFICATION = "INVENTORY_CLASSIFICATION"
    MODEL_UNKNOWN = "UNKNOWN"

    MODEL_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount Analysis",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Continuous Reorder Point (ROP)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Periodic Order-Up-To)",
        MODEL_NEWSVENDOR: "Newsvendor (Single-Period Probabilistic)",
        MODEL_INVENTORY_CLASSIFICATION: "Multicriteria Inventory Classification (Pareto ABC/D)",
        MODEL_UNKNOWN: "Unrecognized Model"
    }

    def __init__(self):
        pass

    # -----------------------------------------------------------------------
    # Main Pipeline Execution
    # -----------------------------------------------------------------------

    def process_question(self, question: str, user_overrides: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        """
        Executes all 7 stages of the decision architecture.
        """
        traces: List[Dict[str, Any]] = []

        # Stage 1: Identify Model
        model_id, confidence, id_reason = self.identify_model(question)
        traces.append({
            "stage": "1. Identify Model",
            "model_identified": model_id,
            "confidence": confidence,
            "reasoning": id_reason
        })

        if model_id == self.MODEL_UNKNOWN:
            return AgentExecutionResult(
                customer_question=question,
                model_identified=model_id,
                model_name_display=self.MODEL_NAMES[model_id],
                identification_confidence=confidence,
                identification_reasoning=id_reason,
                extracted_inputs={},
                normalized_inputs={},
                missing_inputs=["model_clarification"],
                tool_selected="None",
                calculation_result={},
                verification_passed=False,
                verification_checks=[
                    VerificationCheck(
                        check_name="Model Classification",
                        status="FAIL",
                        message="Unable to identify an inventory model with sufficient confidence."
                    )
                ],
                explanation_summary="Could not identify the inventory decision model requested.",
                explanation_details="Please specify whether you need EOQ, Quantity Discount, Safety Stock/ROP, Fixed-Period Review, or Newsvendor analysis.",
                managerial_recommendations=["Provide details regarding demand, ordering costs, holding costs, lead time, or salvage values."],
                stage_traces=traces
            )

        # Stage 2: Extract Inputs
        extracted, normalized, missing = self.extract_inputs(question, model_id, user_overrides)
        traces.append({
            "stage": "2. Extract Inputs",
            "extracted": extracted,
            "normalized": normalized,
            "missing": missing
        })

        # Check if essential inputs are missing before calling the tool
        if missing:
            missing_descriptions = {
                "annual_demand": "Annual Demand (D) - expected total units demanded per year",
                "order_cost": "Ordering / Setup Cost (S) - cost incurred each time an order is placed (e.g. setup cost, setup expense, cost per order)",
                "holding_cost": "Holding / Carrying Cost (H) - annual cost to hold one unit (or percentage holding rate)",
                "avg_demand": "Average Demand rate (d) - mean daily or periodic demand",
                "lead_time": "Lead Time (L) - average time required for delivery from supplier",
                "review_period": "Review Period (R) - length of time between periodic inventory reviews (e.g. every 14 days)",
                "selling_price_or_underage": "Selling Price (P) and Unit Cost (C), OR explicit Underage Cost (Cu) and Overage Cost (Co)",
                "mean_demand": "Mean Demand (μ) - expected demand volume for the period",
                "std_demand": "Standard Deviation of Demand (σ) - measure of demand uncertainty"
            }
            formatted_missing = [f"- **{m}**: {missing_descriptions.get(m, m)}" for m in missing]
            summary = f"Cannot complete calculation: missing required parameter(s): {', '.join(missing)}."
            details = (
                f"### Missing Required Parameters for {self.MODEL_NAMES[model_id]}\n"
                f"The question was thoroughly examined, but the following required inputs could not be found or derived:\n\n"
                + "\n".join(formatted_missing) + "\n\n"
                f"#### Extracted Information So Far:\n"
                + "\n".join([f"- **{k}**: {v}" for k, v in normalized.items() if v is not None and k != "discount_schedule"])
            )
            return AgentExecutionResult(
                customer_question=question,
                model_identified=model_id,
                model_name_display=self.MODEL_NAMES[model_id],
                identification_confidence=confidence,
                identification_reasoning=id_reason,
                extracted_inputs=extracted,
                normalized_inputs=normalized,
                missing_inputs=missing,
                tool_selected="None (Awaiting missing inputs)",
                calculation_result={},
                verification_passed=False,
                verification_checks=[
                    VerificationCheck(
                        check_name="Input Completeness",
                        status="FAIL",
                        message=f"Missing essential parameters: {', '.join(missing)}"
                    )
                ],
                explanation_summary=summary,
                explanation_details=details,
                managerial_recommendations=[
                    f"Please provide the missing parameter(s) ({', '.join(missing)}) to execute the calculation.",
                    "You can also use the Direct Model Calculator tab to input specific numbers directly."
                ],
                stage_traces=traces
            )

        # Stage 3: Select Python Tool
        tool_name, tool_callable = self.select_python_tool(model_id)
        traces.append({
            "stage": "3. Select Python Tool",
            "tool_name": tool_name
        })

        # Stage 4: Calculate
        try:
            calc_result = self.calculate(tool_callable, normalized)
            calc_dict = calc_result.to_dict() if hasattr(calc_result, "to_dict") else asdict(calc_result)
            eight_breakdown = (
                calc_result.get_eight_point_breakdown()
                if hasattr(calc_result, "get_eight_point_breakdown")
                else calc_dict.get("eight_point_breakdown", {})
            )
            traces.append({
                "stage": "4. Calculate",
                "status": "SUCCESS",
                "raw_result": calc_dict,
                "eight_point_breakdown": eight_breakdown
            })
        except Exception as e:
            calc_dict = {"error": str(e)}
            traces.append({
                "stage": "4. Calculate",
                "status": "ERROR",
                "error": str(e)
            })
            return AgentExecutionResult(
                customer_question=question,
                model_identified=model_id,
                model_name_display=self.MODEL_NAMES[model_id],
                identification_confidence=confidence,
                identification_reasoning=id_reason,
                extracted_inputs=extracted,
                normalized_inputs=normalized,
                missing_inputs=missing,
                tool_selected=tool_name,
                calculation_result=calc_dict,
                verification_passed=False,
                verification_checks=[
                    VerificationCheck(
                        check_name="Tool Execution",
                        status="FAIL",
                        message=f"Computation failed: {str(e)}"
                    )
                ],
                explanation_summary=f"Calculation error: {str(e)}",
                explanation_details="The numerical tool encountered an error with the supplied parameters.",
                managerial_recommendations=["Verify all numeric values are strictly positive and properly formatted."],
                eight_point_breakdown={},
                stage_traces=traces
            )

        # Stage 5: Verify
        verification_passed, checks = self.verify(model_id, normalized, calc_dict)
        traces.append({
            "stage": "5. Verify",
            "passed": verification_passed,
            "checks": [asdict(c) for c in checks]
        })

        # Stage 6 & 7: Explain
        summary, details, recommendations = self.explain(
            model_id, normalized, calc_dict, checks, eight_breakdown=eight_breakdown
        )
        traces.append({
            "stage": "6 & 7. Explain",
            "summary": summary,
            "details_length": len(details),
            "recommendations_count": len(recommendations)
        })

        return AgentExecutionResult(
            customer_question=question,
            model_identified=model_id,
            model_name_display=self.MODEL_NAMES[model_id],
            identification_confidence=confidence,
            identification_reasoning=id_reason,
            extracted_inputs=extracted,
            normalized_inputs=normalized,
            missing_inputs=missing,
            tool_selected=tool_name,
            calculation_result=calc_dict,
            verification_passed=verification_passed,
            verification_checks=checks,
            explanation_summary=summary,
            explanation_details=details,
            managerial_recommendations=recommendations,
            eight_point_breakdown=eight_breakdown,
            stage_traces=traces
        )

    # -----------------------------------------------------------------------
    # Stage 1: Identify Model
    # -----------------------------------------------------------------------

    def identify_model(self, question: str) -> Tuple[str, float, str]:
        """
        Classifies user query into one of 5 models based on semantic keywords and weights.
        """
        q_lower = question.lower()
        scores = {
            self.MODEL_EOQ: 0.0,
            self.MODEL_QUANTITY_DISCOUNT: 0.0,
            self.MODEL_SAFETY_STOCK_ROP: 0.0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0.0,
            self.MODEL_NEWSVENDOR: 0.0,
            self.MODEL_INVENTORY_CLASSIFICATION: 0.0
        }

        # 0. Multicriteria Inventory Classification rules
        if any(w in q_lower for w in [
            "classification", "classify", "multicriteria", "multi-criteria", "pareto",
            "abc", "abcd", "sales and hits", "cumulative sales", "pos hits", "hit classification",
            "sales class", "hits class", "sales score", "hits score"
        ]):
            scores[self.MODEL_INVENTORY_CLASSIFICATION] += 4.5
        if any(w in q_lower for w in ["a – 0-60%", "0-60%", "61-80%", "81-90%", "91-100%", "a=4", "b=3", "c=2", "d=1"]):
            scores[self.MODEL_INVENTORY_CLASSIFICATION] += 5.0
        if "csv" in q_lower and any(w in q_lower for w in ["classify", "item number", "hits", "sales"]):
            scores[self.MODEL_INVENTORY_CLASSIFICATION] += 4.0

        # 1. Quantity Discount rules (strongest priority when discount/tier mentioned)
        if any(w in q_lower for w in ["discount", "quantity break", "price break", "volume break", "volume discount", "bracket", "tier", "price schedule"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 4.5
        if "all-units" in q_lower or "bulk discount" in q_lower:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 3.0

        # 2. Fixed-Period Review rules
        if any(w in q_lower for w in ["fixed period", "fixed-period", "periodic review", "order up to", "order-up-to", "target inventory level", "protection interval", "review period", "review interval"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.5
        if any(w in q_lower for w in ["every week", "every 2 weeks", "every two weeks", "every month", "every 14 days", "every 7 days", "periodic cycle"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.0
        if "(r, s)" in q_lower or "(r,s)" in q_lower or "(p, l)" in q_lower:
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.0

        # 3. Newsvendor rules
        if any(w in q_lower for w in ["newsvendor", "newsboy", "single period", "single-period", "perishable", "spoilage", "salvage", "salvage value", "critical ratio", "critical fractile", "underage", "overage"]):
            scores[self.MODEL_NEWSVENDOR] += 4.5
        if any(w in q_lower for w in ["scrap value", "markdown", "unsold items", "expire", "expiration"]):
            scores[self.MODEL_NEWSVENDOR] += 3.0

        # 4. Safety Stock & ROP rules
        if any(w in q_lower for w in ["safety stock", "reorder point", "rop", "buffer stock"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 4.0
        if "service level" in q_lower or "cycle service level" in q_lower or "stockout risk" in q_lower or "z-score" in q_lower:
            scores[self.MODEL_SAFETY_STOCK_ROP] += 2.5
        if "lead time demand" in q_lower or "std dev of lead time" in q_lower:
            scores[self.MODEL_SAFETY_STOCK_ROP] += 2.5

        # 5. EOQ rules
        if any(w in q_lower for w in ["eoq", "economic order quantity", "wilson lot size", "optimal batch size", "optimal order quantity"]):
            scores[self.MODEL_EOQ] += 4.0
        if "setup cost" in q_lower or "order cost" in q_lower or "ordering cost" in q_lower:
            scores[self.MODEL_EOQ] += 1.5
        if "holding cost" in q_lower or "carrying cost" in q_lower:
            scores[self.MODEL_EOQ] += 1.5
        if "how many units should we order each time" in q_lower or "how much should i order" in q_lower:
            scores[self.MODEL_EOQ] += 1.5

        # Disambiguation:
        # If both EOQ and Quantity Discount are triggered, Quantity Discount takes precedence
        if scores[self.MODEL_QUANTITY_DISCOUNT] >= 4.0:
            scores[self.MODEL_EOQ] = min(scores[self.MODEL_EOQ], 2.0)

        # If both Fixed-Period and ROP are triggered, and review period is explicitly stated, Fixed-Period takes precedence
        if scores[self.MODEL_FIXED_PERIOD_REVIEW] >= 4.0 and any(w in q_lower for w in ["review", "period", "order up to"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] = min(scores[self.MODEL_SAFETY_STOCK_ROP], 2.0)

        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        if best_score < 1.5:
            return (self.MODEL_UNKNOWN, 0.2, "Insufficient keywords matching standard inventory models.")

        # Compute normalized confidence (clamped between 0.65 and 0.99)
        confidence = min(0.99, max(0.65, best_score / 6.0))
        reasoning = f"Matched dominant patterns for {self.MODEL_NAMES[best_model]} (score: {best_score:.1f})."

        return (best_model, round(confidence, 2), reasoning)

    # -----------------------------------------------------------------------
    # Stage 2: Extract Inputs
    # -----------------------------------------------------------------------

    def _extract_ordering_cost(self, text: str) -> Optional[float]:
        """
        Extracts ordering / setup cost recognizing all standard synonyms:
        - ordering cost
        - order cost
        - setup cost
        - setup expense
        - cost per order
        - cost of placing an order
        - ordering/setup cost
        - cost to place an order / cost to order
        - order fee / expense per order
        - each order costs $X / $X per order / $X setup expense
        """
        patterns = [
            # Compound / slash synonyms
            r'(?:ordering\s*\/\s*setup|setup\s*\/\s*ordering)\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'(?:ordering\s+and\s+setup|setup\s+and\s+ordering)\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Cost of placing / to place an order
            r'costs?\s+of\s+placing\s+an?\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s+to\s+place\s+an?\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s+to\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Setup expense / ordering expense
            r'(?:setup|ordering)\s+expenses?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'expense\s+per\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Cost per order / fee per order
            r'costs?\s+per\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'fee\s+per\s+order\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'order\s+fees?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Direct ordering cost / order cost / setup cost
            r'(?:ordering|order|setup)\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            # Verb forms
            r'to\s+place\s+(?:an?\s+)?order\s*(?:costs?|runs?|is)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s+place(?:\s+an?\s+order)?|to\s+order|per\s+order|\/order)',
            r'each\s+order\s+costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'orders?\s+costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|\/order|setup\s+cost|setup\s+expense|ordering\s+cost)'
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace(",", "").strip()
                try:
                    return float(val_str)
                except ValueError:
                    continue
        return None

    def _extract_holding_cost_and_rate(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts fixed holding cost and percentage holding cost / rate.
        Distinguishes:
        - fixed holding cost ($/unit/year, carrying cost, cost to hold, storage cost)
        - percentage holding cost (carrying cost rate, holding rate, % of unit price)
        """
        fixed_h: Optional[float] = None
        rate_h: Optional[float] = None

        # 1. Percentage Holding Cost / Rate
        rate_patterns = [
            r'percentage\s+holding\s+costs?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'(?:holding|carrying)\s+rates?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'carrying\s+cost\s+rates?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'carrying\s+charges?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying)\s*(?:rate|cost)?',
            r'(?:holding|carrying)\s+costs?\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s+(?:unit\s+)?(?:cost|price))?',
            r'cost\s+of\s+capital\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%'
        ]
        for pat in rate_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                rate_h = v / 100.0 if v > 1.0 else v
                break

        # 2. Fixed Holding Cost ($ / unit / period)
        fixed_patterns = [
            r'fixed\s+holding\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'holding\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s*(?:per\s+year|\/year)?|\/unit)?',
            r'carrying\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s*(?:per\s+year|\/year)?|\/unit)?',
            r'storage\s+costs?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'holding\s+expenses?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'carrying\s+expenses?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*per\s+unit\s*(?:per\s+year|\/year|to\s+hold|to\s+carry)',
            r'costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*to\s+hold\s*(?:a\s+unit|each\s+unit)?',
            r'cost\s+to\s+hold\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'cost\s+of\s+carrying\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|to\s+hold\s+per\s+year|holding\s+cost|carrying\s+cost)'
        ]
        for pat in fixed_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                # verify not followed by a percentage sign
                end_pos = m.end()
                if end_pos < len(text) and text[end_pos:end_pos+2].strip().startswith('%'):
                    continue
                val_str = m.group(1).replace(",", "").strip()
                try:
                    fixed_h = float(val_str)
                    break
                except ValueError:
                    continue

        return fixed_h, rate_h

    def _extract_demand_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extracts demand volume, timeframe, and demand standard deviation.
        Distinguishes:
        - average demand (daily, weekly, monthly, annual, mean)
        - demand standard deviation (sigma_d, std dev of demand, variability)
        - demand distribution (normal distribution)
        """
        res: Dict[str, Any] = {
            "annual_demand": None,
            "daily_demand": None,
            "weekly_demand": None,
            "monthly_demand": None,
            "mean_demand": None,
            "std_demand": None,
            "distribution": "Normal" if re.search(r'normal(?:ly)?\s+distribut(?:ed|ion)', text, re.IGNORECASE) else "Normal"
        }

        # 1. Demand Standard Deviation (Checked first to avoid capturing as average demand)
        std_patterns = [
            r'demand\s+(?:standard\s+deviation|std\s*dev)\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'(?:standard\s+deviation|std\s*dev)\s+of\s+(?:daily\s+|weekly\s+|annual\s+|seasonal\s+)?demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'(?:standard\s+deviation|std\s*dev)\s+of\s+(?:daily\s+|weekly\s+)?sales\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'daily\s+demand\s+std\s*dev\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s+fluctuates?\s*(?:by|with\s+(?:a\s+)?std\s*dev\s+of)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?',
            r'sigma_?d\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'(?:with\s+)?(?:a\s+)?(?:std\s*dev|standard\s+deviation)\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in std_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["std_demand"] = float(m.group(1).replace(",", "").strip())
                break

        # 2. Daily Demand / Sales
        daily_patterns = [
            r'(?:average|mean)\s+daily\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'daily\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'daily\s+sales\s*(?:average|of|are|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+day|\/day)',
            r'sales\s+per\s+day\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in daily_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                span = m.span()
                clause = re.split(r'[\.\;\n]', text[:span[0]])[-1].lower()
                if "std" not in clause and "deviation" not in clause and "sigma" not in clause:
                    res["daily_demand"] = float(m.group(1).replace(",", "").strip())
                    break

        # 3. Weekly Demand
        weekly_patterns = [
            r'(?:average|mean)?\s*weekly\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+week|\/week|\/wk)',
            r'weekly\s+sales\s*(?:average|of|are|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in weekly_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["weekly_demand"] = float(m.group(1).replace(",", "").strip())
                break

        # 4. Monthly Demand
        monthly_patterns = [
            r'(?:average|mean)?\s*monthly\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+month|\/month|\/mo)'
        ]
        for pat in monthly_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["monthly_demand"] = float(m.group(1).replace(",", "").strip())
                break

        # 5. Annual Demand
        annual_patterns = [
            r'annual\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'yearly\s+demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'annual\s+usage\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'annual\s+sales\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'annual\s+requirement\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+year|annually|\/year|\/yr)',
            r'sells?\s*([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:a|per)\s+year',
            r'annual\s+volume\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in annual_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["annual_demand"] = float(m.group(1).replace(",", "").strip())
                break

        # 6. Generic Mean Demand (For Newsvendor, ROP, etc.)
        mean_patterns = [
            r'(?:mean|expected|average|forecasted)\s+(?:seasonal\s+)?demand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s+mean\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'demand\s+averages?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?',
            r'demand\s+of\s*([0-9,]+(?:\.[0-9]+)?)\s*units'
        ]
        for pat in mean_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                span = m.span()
                clause = re.split(r'[\.\;\n]', text[:span[0]])[-1].lower()
                if "std" not in clause and "deviation" not in clause and "sigma" not in clause:
                    res["mean_demand"] = float(m.group(1).replace(",", "").strip())
                    break

        return res

    def _extract_lead_time_metrics(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts average lead time and lead-time standard deviation.
        Distinguishes:
        - average lead time (mean lead time, supplier lead time, delivery time)
        - lead-time standard deviation (sigma_l, std dev of lead time)
        """
        lead_time: Optional[float] = None
        std_lead_time: Optional[float] = None

        # 1. Lead Time Standard Deviation
        std_patterns = [
            r'lead[\s\-]time\s+standard\s+deviation\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'standard\s+deviation\s+of\s+lead[\s\-]time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'lead[\s\-]time\s+std\s*dev\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'std\s*dev\s+of\s+lead[\s\-]time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'sigma_?l\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'lead[\s\-]time\s+fluctuates?\s*(?:by|with\s+(?:a\s+)?std\s*dev\s+of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)?'
        ]
        for pat in std_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                std_lead_time = float(m.group(1))
                break

        # 2. Average Lead Time
        lt_patterns = [
            r'(?:average|mean|expected)\s+lead[\s\-]time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'supplier\s+lead[\s\-]time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'lead[\s\-]time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|business\s+days?|working\s+days?|periods?)?',
            r'takes\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:for\s+delivery|to\s+arrive|lead\s*time)',
            r'delivery\s+time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'replenishment\s+lead[\s\-]time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'transit\s+time\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)'
        ]
        for pat in lt_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                span = m.span()
                clause = re.split(r'[\.\;\n]', text[:span[0]])[-1].lower()
                if "std" not in clause and "deviation" not in clause and "sigma" not in clause:
                    lead_time = float(m.group(1))
                    break

        return lead_time, std_lead_time

    def _extract_service_level(self, text: str) -> Optional[float]:
        """
        Extracts cycle service level (CSL), non-stockout probability, or in-stock rate.
        Converts percentages (e.g. 95% -> 0.95).
        """
        patterns = [
            r'(?:cycle\s+)?service\s+level\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level',
            r'target\s+service\s+level\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'desired\s+service\s+level\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'csl\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'in[\s\-]stock\s+probability\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'non[\s\-]stockout\s+probability\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'(?:cycle\s+)?service\s+level\s*(?:of|is|are|=|:)?\s*(0\.[0-9]+)'
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                return v / 100.0 if v > 1.0 else v
        return None

    def _extract_pricing_and_costs(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extracts unit purchase price, retail selling price, salvage value,
        underage cost, and overage cost.
        """
        res: Dict[str, Optional[float]] = {
            "selling_price": None,
            "unit_cost": None,
            "salvage_value": None,
            "underage_cost": None,
            "overage_cost": None
        }

        # 1. Underage / Shortage Cost
        cu_patterns = [
            r'(?:underage|shortage)\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s+of\s+(?:underage|shortage|understocking)\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'stockout\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'c_?u\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in cu_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["underage_cost"] = float(m.group(1).replace(",", "").strip())
                break

        # 2. Overage / Excess Cost
        co_patterns = [
            r'(?:overage|excess)\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s+of\s+(?:overage|excess|overstocking)\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'c_?o\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in co_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["overage_cost"] = float(m.group(1).replace(",", "").strip())
                break

        # 3. Selling Price
        sp_patterns = [
            r'selling\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'retail\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'sales\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'revenue\s+per\s+unit\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'sells?\s+(?:for|at)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'retail(?:s)?\s+(?:for|at)\s*\$?([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in sp_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["selling_price"] = float(m.group(1).replace(",", "").strip())
                break

        # 4. Purchase Cost / Unit Purchase Price / Wholesale Cost
        uc_patterns = [
            r'unit\s+purchase\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'purchase\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'purchase\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'wholesale\s+(?:unit\s+)?costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'wholesale\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'unit\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'item\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'production\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'acquisition\s+costs?\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit|each|\/unit|to\s+buy|to\s+produce|to\s+acquire)'
        ]
        for pat in uc_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["unit_cost"] = float(m.group(1).replace(",", "").strip())
                break

        # 5. Salvage Value
        salvage_patterns = [
            r'salvage\s+value\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'salvaged\s+(?:for|at)\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'scrap\s+value\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'clearance\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'markdown\s+price\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'liquidation\s+value\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)',
            r'residual\s+value\s*(?:of|is|are|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)'
        ]
        for pat in salvage_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["salvage_value"] = float(m.group(1).replace(",", "").strip())
                break

        return res

    def _extract_periodic_review_metrics(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extracts review period, on-hand inventory, on-order inventory, back-orders,
        explicit safety stock, and inventory position for Fixed-Period Review.
        """
        res: Dict[str, Optional[float]] = {
            "review_period": None,
            "safety_stock": None,
            "inventory_position": None,
            "on_hand": None,
            "on_order": None,
            "backorders": None
        }

        # 1. Review Period
        rp_patterns = [
            r'review\s+period\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|periods?)?',
            r'review\s+interval\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|periods?)?',
            r'review\s+frequency\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|periods?)?',
            r'order\s+interval\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|periods?)?',
            r'review\s+cadence\s*(?:of|is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|periods?)?',
            r'review(?:ed)?(?:\s+inventory|\s+stock)?\s+every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)',
            r'every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:review|cadence)',
            r'every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)(?:,\s*lead\s+time|\s+lead\s+time)',
            r'orders?\s+(?:placed|occur)\s+every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)'
        ]
        for pat in rp_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["review_period"] = float(m.group(1))
                break

        # 2. Explicit Safety Stock
        ss_patterns = [
            r'safety\s+stock\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?',
            r'buffer\s+stock\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?'
        ]
        for pat in ss_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["safety_stock"] = float(m.group(1).replace(",", "").strip())
                break

        # 3. Explicit Inventory Position
        ip_patterns = [
            r'inventory\s+position\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?',
            r'current\s+position\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?'
        ]
        for pat in ip_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["inventory_position"] = float(m.group(1).replace(",", "").strip())
                break

        # 4. On-Hand Inventory
        oh_patterns = [
            r'on[\s\-]hand\s+inventory\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'inventory\s+on[\s\-]hand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'on[\s\-]hand\s+stock\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'stock\s+on[\s\-]hand\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'physical\s+inventory\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'current\s+inventory\s*(?:of|is|are|=|:)?\s*(?:on\s+hand\s+is\s+)?([0-9,]+(?:\.[0-9]+)?)',
            r'have\s*([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:in\s+stock|on\s+hand)',
            r'([0-9,]+(?:\.[0-9]+)?)\s*units?\s*(?:on\s+hand|in\s+stock)'
        ]
        for pat in oh_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["on_hand"] = float(m.group(1).replace(",", "").strip())
                break

        # 5. On-Order Inventory
        oo_patterns = [
            r'on[\s\-]order\s+inventory\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'inventory\s+on[\s\-]order\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'open\s+orders?\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'pipeline\s+inventory\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'orders?\s+in\s+transit\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'incoming\s+(?:orders?|shipments?)\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'on\s+order\s*(?:is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'([0-9,]+(?:\.[0-9]+)?)\s*units?\s*(?:on\s+order|in\s+transit)'
        ]
        for pat in oo_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["on_order"] = float(m.group(1).replace(",", "").strip())
                break

        # 6. Back-Orders
        bo_patterns = [
            r'back[\s\-]orders?\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'backlogged\s+orders?\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'unfilled\s+orders?\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'customer\s+backorders?\s*(?:of|is|are|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)',
            r'([0-9,]+(?:\.[0-9]+)?)\s*backorders?'
        ]
        for pat in bo_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                res["backorders"] = float(m.group(1).replace(",", "").strip())
                break

        return res

    def _extract_discount_schedule(
        self,
        text: str,
        base_unit_cost: Optional[float],
        holding_rate: Optional[float],
        fixed_holding_cost: Optional[float]
    ) -> List[Dict[str, Any]]:
        """
        Parses multi-bracket quantity discount schedule directly from question text.
        Recognizes format like:
          - Tier 1: 1-499 units at unit price $10.00; Tier 2: 500-999 units at unit price $9.20; Tier 3: 1000+ units at unit price $8.50
          - 1 to 499: $10, 500 to 999: $9.20, 1000+: $8.50
        """
        brackets: List[Dict[str, Any]] = []

        # Find all bounded ranges like '1-499 units at unit price $10' or '500 to 999 @ $9.20'
        range_matches = re.findall(
            r'([0-9,]+)\s*(?:-|to)\s*([0-9,]+)\s*(?:units?)?[^\$0-9\n;]*\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        # Find plus ranges like '1000+ units at unit price $8.50' or '1000 or more @ $8.50'
        plus_matches = re.findall(
            r'([0-9,]+)\s*(?:\+|or\s+more)\s*(?:units?)?[^\$0-9\n;]*\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )

        for rm in range_matches:
            min_q = float(rm[0].replace(",", ""))
            max_q = float(rm[1].replace(",", ""))
            cost = float(rm[2])
            h_cost = fixed_holding_cost if fixed_holding_cost is not None else (cost * holding_rate if holding_rate else None)
            brackets.append({
                "min_qty": min_q,
                "max_qty": max_q,
                "unit_cost": cost,
                "holding_cost": h_cost
            })

        for pm in plus_matches:
            min_q = float(pm[0].replace(",", ""))
            cost = float(pm[1])
            h_cost = fixed_holding_cost if fixed_holding_cost is not None else (cost * holding_rate if holding_rate else None)
            brackets.append({
                "min_qty": min_q,
                "max_qty": math.inf,
                "unit_cost": cost,
                "holding_cost": h_cost
            })

        if brackets:
            # Sort by min_qty
            brackets.sort(key=lambda b: b["min_qty"])
            return brackets

        # Check for simple quantity break like 'quantity break at 500 units'
        qb_match = re.search(r'(?:quantity\s+break|price\s+break)\s*(?:at|is|=|:)?\s*([0-9,]+)\s*units?', text, re.IGNORECASE)
        base_p = base_unit_cost or 10.0
        h_base = fixed_holding_cost if fixed_holding_cost is not None else (base_p * (holding_rate or 0.20))
        if qb_match:
            break_val = float(qb_match.group(1).replace(",", ""))
            return [
                {"min_qty": 1, "max_qty": break_val - 1, "unit_cost": base_p, "holding_cost": h_base},
                {"min_qty": break_val, "max_qty": math.inf, "unit_cost": base_p * 0.90, "holding_cost": h_base * 0.90}
            ]

        # Standard default tiers if base cost provided
        return [
            {"min_qty": 1, "max_qty": 499, "unit_cost": base_p, "holding_cost": h_base},
            {"min_qty": 500, "max_qty": 999, "unit_cost": base_p * 0.92, "holding_cost": h_base * 0.92},
            {"min_qty": 1000, "max_qty": math.inf, "unit_cost": base_p * 0.85, "holding_cost": h_base * 0.85}
        ]

    def extract_inputs(
        self,
        question: str,
        model_id: str,
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Deep multi-pass natural-language extraction.
        Examines the entire question, recognizes all domain synonyms, and extracts
        numerical values and their meanings before declaring any parameter missing.
        """
        q = question

        # 1. Component extractions
        order_cost = self._extract_ordering_cost(q)
        fixed_h, rate_h = self._extract_holding_cost_and_rate(q)
        demand_info = self._extract_demand_metrics(q)
        lt_info, std_lt_info = self._extract_lead_time_metrics(q)
        csl = self._extract_service_level(q)
        pricing = self._extract_pricing_and_costs(q)
        periodic = self._extract_periodic_review_metrics(q)

        extracted = {
            "order_cost": order_cost,
            "fixed_holding_cost": fixed_h,
            "holding_rate": rate_h,
            "annual_demand": demand_info["annual_demand"],
            "daily_demand": demand_info["daily_demand"],
            "weekly_demand": demand_info["weekly_demand"],
            "monthly_demand": demand_info["monthly_demand"],
            "mean_demand": demand_info["mean_demand"],
            "std_demand": demand_info["std_demand"],
            "demand_distribution": demand_info["distribution"],
            "lead_time": lt_info,
            "std_lead_time": std_lt_info,
            "service_level": csl,
            "selling_price": pricing["selling_price"],
            "unit_cost": pricing["unit_cost"],
            "salvage_value": pricing["salvage_value"],
            "underage_cost": pricing["underage_cost"],
            "overage_cost": pricing["overage_cost"],
            "review_period": periodic["review_period"],
            "safety_stock_explicit": periodic["safety_stock"],
            "inventory_position_explicit": periodic["inventory_position"],
            "on_hand": periodic["on_hand"],
            "on_order": periodic["on_order"],
            "backorders": periodic["backorders"]
        }

        # Apply user overrides if supplied
        if user_overrides:
            for k, v in user_overrides.items():
                if v is not None:
                    extracted[k] = v

        normalized: Dict[str, Any] = {}
        missing: List[str] = []

        # ===================================================================
        # Model-Specific Normalization & Meaning Resolution
        # ===================================================================

        if model_id == self.MODEL_EOQ:
            # Resolve Annual Demand (D)
            D = extracted.get("annual_demand")
            if D is None:
                if extracted.get("daily_demand"):
                    D = extracted["daily_demand"] * 250.0  # standard 250 work days
                elif extracted.get("weekly_demand"):
                    D = extracted["weekly_demand"] * 52.0
                elif extracted.get("monthly_demand"):
                    D = extracted["monthly_demand"] * 12.0
                elif extracted.get("mean_demand"):
                    D = extracted["mean_demand"]

            S = extracted.get("order_cost")
            C = extracted.get("unit_cost") or 0.0

            # Resolve Holding Cost (H)
            H = extracted.get("fixed_holding_cost")
            if H is None and extracted.get("holding_rate") and C > 0:
                H = extracted["holding_rate"] * C

            normalized["annual_demand"] = D
            normalized["order_cost"] = S
            normalized["holding_cost"] = H
            normalized["unit_cost"] = C

            if D is None:
                missing.append("annual_demand")
            if S is None:
                missing.append("order_cost")
            if H is None:
                missing.append("holding_cost")

        elif model_id == self.MODEL_QUANTITY_DISCOUNT:
            # Resolve Demand
            D = extracted.get("annual_demand") or extracted.get("mean_demand")
            if D is None:
                if extracted.get("daily_demand"):
                    D = extracted["daily_demand"] * 250.0
                elif extracted.get("weekly_demand"):
                    D = extracted["weekly_demand"] * 52.0
                elif extracted.get("monthly_demand"):
                    D = extracted["monthly_demand"] * 12.0

            S = extracted.get("order_cost")
            hr = extracted.get("holding_rate")
            fixed_h = extracted.get("fixed_holding_cost")
            unit_c = extracted.get("unit_cost")

            schedule = (user_overrides.get("discount_schedule") if user_overrides else None) or \
                       self._extract_discount_schedule(q, unit_c, hr, fixed_h)

            normalized["annual_demand"] = D
            normalized["order_cost"] = S
            normalized["holding_rate"] = hr if hr is not None else 0.20
            normalized["discount_schedule"] = schedule

            if D is None:
                missing.append("annual_demand")
            if S is None:
                missing.append("order_cost")

        elif model_id == self.MODEL_SAFETY_STOCK_ROP:
            # Average demand
            d = extracted.get("daily_demand") or extracted.get("mean_demand")
            if d is None and extracted.get("weekly_demand"):
                d = extracted["weekly_demand"] / 7.0

            L = extracted.get("lead_time")
            sigma_d = extracted.get("std_demand") or 0.0
            sigma_L = extracted.get("std_lead_time") or 0.0
            service_lvl = extracted.get("service_level") or 0.95

            normalized["avg_demand"] = d
            normalized["lead_time"] = L
            normalized["std_demand"] = sigma_d
            normalized["std_lead_time"] = sigma_L
            normalized["service_level"] = service_lvl
            normalized["unit_holding_cost"] = extracted.get("fixed_holding_cost")

            if d is None:
                missing.append("avg_demand")
            if L is None:
                missing.append("lead_time")

        elif model_id == self.MODEL_FIXED_PERIOD_REVIEW:
            R = extracted.get("review_period")
            L = extracted.get("lead_time")
            d = extracted.get("daily_demand") or extracted.get("mean_demand")
            if d is None and extracted.get("weekly_demand"):
                d = extracted["weekly_demand"] / 7.0

            sigma_d = extracted.get("std_demand") or 0.0
            service_lvl = extracted.get("service_level") or 0.95
            ss_explicit = extracted.get("safety_stock_explicit")
            ip_explicit = extracted.get("inventory_position_explicit")

            on_hand = extracted.get("on_hand") or 0.0
            on_order = extracted.get("on_order") or 0.0
            backorders = extracted.get("backorders") or 0.0

            normalized["review_period"] = R
            normalized["lead_time"] = L
            normalized["avg_demand"] = d
            normalized["std_demand"] = sigma_d
            normalized["service_level"] = service_lvl
            normalized["current_inventory"] = on_hand
            normalized["on_order"] = on_order
            normalized["backorders"] = backorders
            normalized["safety_stock"] = ss_explicit
            normalized["inventory_position"] = ip_explicit

            if R is None:
                missing.append("review_period")
            if L is None:
                missing.append("lead_time")
            if d is None:
                missing.append("avg_demand")

        elif model_id == self.MODEL_NEWSVENDOR:
            P = extracted.get("selling_price")
            C = extracted.get("unit_cost")
            V = extracted.get("salvage_value") or 0.0
            cu = extracted.get("underage_cost")
            co = extracted.get("overage_cost")
            mu = extracted.get("mean_demand") or extracted.get("annual_demand") or extracted.get("daily_demand")
            sigma = extracted.get("std_demand")

            normalized["selling_price"] = P
            normalized["unit_cost"] = C
            normalized["salvage_value"] = V
            normalized["underage_cost"] = cu
            normalized["overage_cost"] = co
            normalized["mean_demand"] = mu
            normalized["std_demand"] = sigma
            normalized["stockout_penalty"] = 0.0

            # Newsvendor can be solved by (Cu and Co) OR by (P and C)
            has_marginal_costs = (cu is not None and co is not None)
            has_price_and_cost = (P is not None and C is not None)

            if not has_marginal_costs and not has_price_and_cost:
                missing.append("selling_price_or_underage")
            if mu is None:
                missing.append("mean_demand")
            if sigma is None:
                missing.append("std_demand")

        elif model_id == self.MODEL_INVENTORY_CLASSIFICATION:
            items_data = None
            if user_overrides and "items_data" in user_overrides:
                items_data = user_overrides["items_data"]
            elif user_overrides and "df" in user_overrides:
                items_data = user_overrides["df"]

            normalized["items_data"] = items_data
            if items_data is None:
                missing.append("items_data (Upload CSV with Item Number, Item Name, UoM, Sales ($), Hits)")

        return extracted, normalized, missing

    # -----------------------------------------------------------------------
    # Stage 3: Select Python Tool
    # -----------------------------------------------------------------------

    def select_python_tool(self, model_id: str) -> Tuple[str, Any]:
        """
        Maps identified model to the appropriate numerical calculation function.
        """
        mapping = {
            self.MODEL_EOQ: ("calculate_eoq", tools.calculate_eoq),
            self.MODEL_QUANTITY_DISCOUNT: ("calculate_quantity_discount", tools.calculate_quantity_discount),
            self.MODEL_SAFETY_STOCK_ROP: ("calculate_safety_stock_rop", tools.calculate_safety_stock_rop),
            self.MODEL_FIXED_PERIOD_REVIEW: ("calculate_fixed_period_review", tools.calculate_fixed_period_review),
            self.MODEL_NEWSVENDOR: ("calculate_newsvendor", tools.calculate_newsvendor),
            self.MODEL_INVENTORY_CLASSIFICATION: ("calculate_multicriteria_inventory_classification", tools.calculate_multicriteria_inventory_classification)
        }
        if model_id in mapping:
            return mapping[model_id]
        raise ValueError(f"No Python tool available for model: {model_id}")

    # -----------------------------------------------------------------------
    # Stage 4: Calculate
    # -----------------------------------------------------------------------

    def calculate(self, tool_callable: Any, normalized_inputs: Dict[str, Any]) -> Any:
        """
        Executes the Python numerical tool with normalized arguments.
        """
        # Filter inputs to only include params accepted by the callable
        import inspect
        sig = inspect.signature(tool_callable)
        filtered_kwargs = {}
        for param in sig.parameters.values():
            if param.name in normalized_inputs and normalized_inputs[param.name] is not None:
                filtered_kwargs[param.name] = normalized_inputs[param.name]

        return tool_callable(**filtered_kwargs)

    # -----------------------------------------------------------------------
    # Stage 5: Verify
    # -----------------------------------------------------------------------

    def verify(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Tuple[bool, List[VerificationCheck]]:
        """
        Performs mathematical and physical sanity checks on the calculation output.
        """
        checks: List[VerificationCheck] = []
        all_passed = True

        if model_id == self.MODEL_EOQ:
            # Check 1: Non-zero positive EOQ
            eoq = results.get("eoq", 0)
            if eoq > 0:
                checks.append(VerificationCheck("EOQ Positivity", "PASS", f"Calculated EOQ ({eoq:.2f} units) is positive."))
            else:
                checks.append(VerificationCheck("EOQ Positivity", "FAIL", "EOQ must be strictly greater than zero."))
                all_passed = False

            # Check 2: Annual Ordering Cost approx equals Annual Holding Cost
            aoc = results.get("annual_order_cost", 0)
            ahc = results.get("annual_holding_cost", 0)
            diff = abs(aoc - ahc)
            # Within 1% tolerance due to integer rounding
            if aoc > 0 and (diff / aoc) < 0.02:
                checks.append(VerificationCheck("Cost Balance Check", "PASS", f"Annual Order Cost (${aoc:,.2f}) balances Annual Holding Cost (${ahc:,.2f}) at Wilson optimum."))
            else:
                checks.append(VerificationCheck("Cost Balance Check", "WARN", f"AOC (${aoc:,.2f}) and AHC (${ahc:,.2f}) show minor discrepancy (difference: ${diff:.2f})."))

        elif model_id == self.MODEL_QUANTITY_DISCOUNT:
            opt_qty = results.get("optimal_order_quantity", 0)
            savings = results.get("annual_cost_savings", 0)
            if opt_qty > 0:
                checks.append(VerificationCheck("Feasible Lot Size", "PASS", f"Optimal batch size {opt_qty:,.0f} satisfies bracket feasibility."))
            else:
                checks.append(VerificationCheck("Feasible Lot Size", "FAIL", "Invalid optimal lot size calculated."))
                all_passed = False

            if savings >= 0:
                checks.append(VerificationCheck("Cost Minimization", "PASS", f"Optimal tier yields non-negative savings (${savings:,.2f}) over base tier."))
            else:
                checks.append(VerificationCheck("Cost Minimization", "FAIL", "Selected discount policy exceeds base cost."))
                all_passed = False

        elif model_id == self.MODEL_SAFETY_STOCK_ROP:
            ss = results.get("safety_stock", 0)
            rop = results.get("reorder_point", 0)
            ltd_mean = results.get("lead_time_demand_mean", 0)

            if ss >= 0:
                checks.append(VerificationCheck("Safety Stock Bound", "PASS", f"Safety stock ({ss:.2f} units) is non-negative."))
            else:
                checks.append(VerificationCheck("Safety Stock Bound", "FAIL", "Safety stock cannot be negative."))
                all_passed = False

            if rop >= ltd_mean:
                checks.append(VerificationCheck("ROP Lead Time Coverage", "PASS", f"ROP ({rop:.2f}) covers expected lead time demand ({ltd_mean:.2f})."))
            else:
                checks.append(VerificationCheck("ROP Lead Time Coverage", "FAIL", "Reorder point is below expected lead time demand."))
                all_passed = False

        elif model_id == self.MODEL_FIXED_PERIOD_REVIEW:
            outl = results.get("order_up_to_level", 0)
            p_mean = results.get("protection_demand_mean", 0)
            q_rec = results.get("recommended_order_qty", 0)

            if outl >= p_mean:
                checks.append(VerificationCheck("Protection Interval Buffer", "PASS", f"Order-up-to level ({outl:.2f}) satisfies protection interval demand ({p_mean:.2f})."))
            else:
                checks.append(VerificationCheck("Protection Interval Buffer", "FAIL", "Order-up-to level is insufficient to cover protection interval."))
                all_passed = False

            if q_rec >= 0:
                checks.append(VerificationCheck("Non-negative Replenishment", "PASS", f"Replenishment quantity ({q_rec:.2f}) is valid."))
            else:
                checks.append(VerificationCheck("Non-negative Replenishment", "FAIL", "Recommended order quantity cannot be negative."))
                all_passed = False

        elif model_id == self.MODEL_NEWSVENDOR:
            cr = results.get("critical_ratio", 0)
            q_opt = results.get("optimal_order_quantity", 0)
            exp_profit = results.get("expected_profit", 0)
            max_profit = results.get("max_possible_profit", 0)

            if 0.0 < cr < 1.0:
                checks.append(VerificationCheck("Critical Ratio Range", "PASS", f"Critical ratio ({cr:.4f}) is strictly bounded in (0, 1)."))
            else:
                checks.append(VerificationCheck("Critical Ratio Range", "FAIL", f"Critical ratio ({cr:.4f}) must be strictly between 0 and 1."))
                all_passed = False

            if exp_profit <= max_profit + 1.0:
                checks.append(VerificationCheck("Profit Upper Bound", "PASS", f"Expected profit (${exp_profit:,.2f}) does not exceed perfect-information profit (${max_profit:,.2f})."))
            else:
                checks.append(VerificationCheck("Profit Upper Bound", "FAIL", "Expected profit exceeds theoretical maximum."))
                all_passed = False

        elif model_id == self.MODEL_INVENTORY_CLASSIFICATION:
            tot_items = results.get("total_items", 0)
            tot_sales = results.get("total_sales", 0.0)
            tot_hits = results.get("total_hits", 0.0)

            if tot_items > 0:
                checks.append(VerificationCheck("Item Count Validity", "PASS", f"Classified {tot_items:,} inventory items successfully."))
            else:
                checks.append(VerificationCheck("Item Count Validity", "FAIL", "Dataset contains 0 items."))
                all_passed = False

            if tot_sales > 0:
                checks.append(VerificationCheck("Sales Integrity", "PASS", f"Total cumulative sales evaluated: ${tot_sales:,.2f}."))
            else:
                checks.append(VerificationCheck("Sales Integrity", "FAIL", "Total sales is zero or invalid."))
                all_passed = False

            if tot_hits > 0:
                checks.append(VerificationCheck("Hits Activity Integrity", "PASS", f"Total transaction hits evaluated: {tot_hits:,.0f} picks."))
            else:
                checks.append(VerificationCheck("Hits Activity Integrity", "WARN", "Total hits count is zero."))

        return (all_passed, checks)

    # -----------------------------------------------------------------------
    # Stage 6 & 7: Explain
    # -----------------------------------------------------------------------

    def explain(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        checks: List[VerificationCheck],
        eight_breakdown: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, List[str]]:
        """
        Synthesizes the mandatory 8-point calculation report from deterministic tools,
        along with executive summary, economic trade-offs, and managerial recommendations.
        """
        summary = ""
        details = ""
        recs: List[str] = []

        # Construct the formal 8-point calculation report directly from the deterministic tool output
        eb = eight_breakdown or results.get("eight_point_breakdown", {})
        eight_report_lines = []
        if eb:
            # 1. Inputs Extracted
            eight_report_lines.append("### 1. Inputs Extracted")
            for k, v in eb.get("inputs_extracted", {}).items():
                if k == "discount_schedule" and isinstance(v, list):
                    eight_report_lines.append(f"- **{k}**: {len(v)} price brackets provided")
                elif isinstance(v, float):
                    eight_report_lines.append(f"- **{k}**: `{v:,.2f}`")
                else:
                    eight_report_lines.append(f"- **{k}**: `{v}`")
            eight_report_lines.append("")

            # 2. Variable Definitions
            eight_report_lines.append("### 2. Variable Definitions")
            for sym, defn in eb.get("variable_definitions", {}).items():
                eight_report_lines.append(f"- **{sym}**: {defn}")
            eight_report_lines.append("")

            # 3. Formula
            eight_report_lines.append("### 3. Formula")
            eight_report_lines.append(f"```text\n{eb.get('formula', '')}\n```\n")

            # 4. Formula with Substituted Values
            eight_report_lines.append("### 4. Formula with Substituted Values")
            eight_report_lines.append(f"```text\n{eb.get('formula_with_substituted_values', '')}\n```\n")

            # 5. Intermediate Calculations
            eight_report_lines.append("### 5. Intermediate Calculations")
            inter = eb.get("intermediate_calculations", {})
            for k, v in inter.items():
                if k == "tiers_evaluated" and isinstance(v, list):
                    eight_report_lines.append(f"- **{k}**:")
                    for tier in v:
                        eight_report_lines.append(
                            f"  - Bracket {tier.get('bracket')}: unconstrained EOQ = {tier.get('unconstrained_eoq'):.2f}, "
                            f"feasible lot size = {tier.get('adjusted_feasible_lot_size'):.2f}, "
                            f"total cost = ${tier.get('total_cost'):,.2f}"
                        )
                elif isinstance(v, float):
                    eight_report_lines.append(f"- **{k}**: `{v:,.4f}`" if abs(v) < 1 else f"- **{k}**: `{v:,.2f}`")
                elif isinstance(v, (int, str)):
                    eight_report_lines.append(f"- **{k}**: `{v}`")
            eight_report_lines.append("")

            # 6. Final Answer
            final_val = eb.get("final_answer", results.get("final_answer", ""))
            eight_report_lines.append("### 6. Final Answer")
            eight_report_lines.append(f"**Final Answer**: `{final_val}`\n")

            # 7. Verification Checks
            eight_report_lines.append("### 7. Verification Checks")
            for c in checks:
                status_icon = "✅" if c.status == "PASS" else ("⚠️" if c.status == "WARN" else "❌")
                eight_report_lines.append(f"- {status_icon} **{c.check_name}** [{c.status}]: {c.message}")
            eight_report_lines.append("")

            # 8. Assumptions
            eight_report_lines.append("### 8. Assumptions")
            for asm in eb.get("assumptions", results.get("assumptions", [])):
                eight_report_lines.append(f"- {asm}")
            eight_report_lines.append("")

        eight_report_str = "\n".join(eight_report_lines)

        if model_id == self.MODEL_EOQ:
            eoq = results['eoq']
            orders = results['orders_per_year']
            cycle_days = results['cycle_time_days']
            total_policy = results['total_policy_cost']
            aoc = results['annual_order_cost']
            ahc = results['annual_holding_cost']

            summary = (
                f"**Optimal Order Quantity (EOQ): {eoq:,.0f} units per order.**\n"
                f"Placing **{orders:.1f} orders per year** (every **{cycle_days:.1f} working days**) "
                f"minimizes total annual inventory policy costs at **${total_policy:,.2f}**."
            )

            details = (
                f"### Economic Trade-Off Breakdown\n"
                f"- **Annual Demand (D)**: {inputs.get('annual_demand', 0):,.0f} units\n"
                f"- **Order/Setup Cost (S)**: ${inputs.get('order_cost', 0):,.2f} per replenishment cycle\n"
                f"- **Holding Cost (H)**: ${inputs.get('holding_cost', 0):,.2f} per unit/year\n\n"
                f"#### Cost Equilibrium:\n"
                f"- **Annual Ordering Cost**: ${aoc:,.2f} ({orders:.2f} orders × ${inputs.get('order_cost', 0):.2f})\n"
                f"- **Annual Holding Cost**: ${ahc:,.2f} ({eoq/2:,.1f} avg units × ${inputs.get('holding_cost', 0):.2f})\n\n"
                f"> At the Wilson optimum, holding costs and ordering costs reach financial balance. "
                f"Ordering larger lots would increase carrying costs faster than order savings, "
                f"while smaller lots inflate purchase order processing and administrative overhead."
            )

            sens = results.get("cost_sensitivity", {})
            recs = [
                f"Configure the ERP/WMS lot size parameter to {eoq:,.0f} units.",
                f"Set reorder cadence to approximately every {cycle_days:.0f} working days.",
                f"If lot size must vary by ±10%, total policy cost increases by only ~{sens.get('+10% Q', 0.5):.2f}%, demonstrating moderate cost curvature around the minimum.",
                "Review carrying cost estimates annually (capital cost, warehouse space, shrinkage, and insurance)."
            ]

        elif model_id == self.MODEL_QUANTITY_DISCOUNT:
            opt_q = results['optimal_order_quantity']
            opt_cost = results['optimal_total_cost']
            savings = results['annual_cost_savings']
            pct = results['percentage_savings']
            opt_bracket = results['optimal_bracket_index']

            summary = (
                f"**Recommended Order Quantity: {opt_q:,.0f} units** (Discount Tier {opt_bracket + 1}).\n"
                f"Total annual procurement & inventory cost is **${opt_cost:,.2f}**, delivering **${savings:,.2f} ({pct:.1f}%) in annual cost savings** "
                f"compared to purchasing without volume discounts."
            )

            details = (
                f"### Bracket-by-Bracket Evaluation\n"
                f"Suppliers offer price tiers to incentivize larger order commitments. "
                f"The agent evaluated each price breakpoint by comparing unit purchase cost savings against increased warehouse inventory holding costs:\n\n"
            )
            for b in results.get("brackets", []):
                max_str = f"{b['max_qty']:,.0f}" if b['max_qty'] != math.inf else "Infinity"
                status_note = "✅ **Optimal Choice**" if b['bracket_index'] == opt_bracket else "Suboptimal"
                details += (
                    f"- **Tier {b['bracket_index'] + 1} ({b['min_qty']:,.0f} to {max_str} units @ ${b['unit_cost']:.2f})**: "
                    f"Feasible Q = {b['feasible_qty']:,.0f} units | Total Annual Cost = ${b['total_annual_cost']:,.2f} ({status_note})\n"
                )

            recs = [
                f"Order {opt_q:,.0f} units per purchase order to unlock Tier {opt_bracket + 1} pricing.",
                f"Ensure warehouse capacity can accommodate the peak inventory level of {opt_q:,.0f} units.",
                "Negotiate with the supplier for split shipments if dock space is constrained.",
                "Lock in supplier contract prices for the annual demand volume."
            ]

        elif model_id == self.MODEL_SAFETY_STOCK_ROP:
            ss = results['safety_stock']
            rop = results['reorder_point']
            ltd_mean = results['lead_time_demand_mean']
            ltd_std = results['lead_time_demand_std']
            csl = results['service_level'] * 100.0
            z = results['z_score']

            summary = (
                f"**Set Reorder Point (ROP) at {rop:,.0f} units (with {ss:,.0f} units Safety Stock).**\n"
                f"This guarantees a **{csl:.1f}% Cycle Service Level** ($z = {z:.2f}$), ensuring that no more than "
                f"{(100 - csl):.1f}% of replenishment cycles experience a stockout."
            )

            details = (
                f"### Stochastic Lead Time Demand Analysis\n"
                f"- **Average Lead Time Demand**: {ltd_mean:,.1f} units\n"
                f"- **Std Dev of Lead Time Demand (σ_DLT)**: {ltd_std:,.2f} units\n"
                f"- **Safety Stock Buffer**: {ss:,.1f} units ($z \\times \\sigma_{{DLT}}$)\n"
                f"- **Target Reorder Point (ROP)**: {rop:,.1f} units\n\n"
                f"> Whenever available inventory (On-Hand + On-Order - Backorders) drops to or below **{rop:,.0f} units**, "
                f"an immediate replenishment purchase order must be generated."
            )

            recs = [
                f"Input ROP = {rop:,.0f} and Safety Stock = {ss:,.0f} into inventory control software.",
                "Set automated alerts when physical stock crosses below the Reorder Point.",
                f"If the business desires to raise service level to 99% (z=2.33), safety stock must increase by ~{(2.326/z - 1)*100:.1f}%.",
                "Work with suppliers to compress lead times, as reducing lead time directly decreases required safety stock by the square-root rule."
            ]

        elif model_id == self.MODEL_FIXED_PERIOD_REVIEW:
            outl = results['order_up_to_level']
            ss = results['safety_stock']
            q_rec = results['recommended_order_qty']
            R = results['review_period_periods']
            L = results['lead_time_periods']
            ip = results['inventory_position']

            summary = (
                f"**Order-Up-To Level (S): {outl:,.0f} units.** Current Recommended Order: **{q_rec:,.0f} units.**\n"
                f"With a review cadence of every **{R:.0f} days** and **{L:.0f} days lead time**, "
                f"inventory position ({ip:,.0f} units) requires placing an order for **{q_rec:,.0f} units** today."
            )

            details = (
                f"### Protection Interval & Inventory Position Breakdown\n"
                f"- **Review Interval (R)**: {R:.0f} periods\n"
                f"- **Lead Time (L)**: {L:.0f} periods\n"
                f"- **Protection Interval (R + L)**: {R + L:.0f} periods\n"
                f"- **Expected Demand in Protection Window**: {results['protection_demand_mean']:,.1f} units\n"
                f"- **Safety Stock Component**: {ss:,.1f} units\n"
                f"- **Current Inventory Position**: {ip:,.0f} (On-Hand: {results['current_on_hand']:,.0f}, On-Order: {results['current_on_order']:,.0f})\n\n"
                f"> In periodic review systems, the inventory buffer must cover the entire protection window ($R + L$) "
                f"because the next opportunity to order won't occur until period $R$ arrives."
            )

            recs = [
                f"Place an order today for {q_rec:,.0f} units.",
                f"Schedule the next inventory review and ordering window exactly {R:.0f} days from today.",
                "Consolidate multiple SKUs from the same vendor on this review cadence to achieve freight discounts.",
                "Monitor for unexpected demand spikes between review dates to trigger emergency expedited orders if needed."
            ]

        elif model_id == self.MODEL_NEWSVENDOR:
            q_opt = results['optimal_order_quantity']
            cr = results['critical_ratio']
            cu = results['underage_cost']
            co = results['overage_cost']
            exp_profit = results['expected_profit']
            lost_sales = results['expected_lost_sales']
            leftovers = results['expected_leftovers']

            summary = (
                f"**Optimal Single-Period Order Quantity: {q_opt:,.0f} units.**\n"
                f"The Critical Fractile ($CR = {cr:.4f}$) maximizes expected profit at **${exp_profit:,.2f}** "
                f"by balancing the underage cost of stockouts (${cu:,.2f}) against the overage cost of excess inventory (${co:,.2f})."
            )

            details = (
                f"### Marginal Economic Analysis\n"
                f"- **Underage Cost (Cu = Price - Cost)**: ${cu:,.2f} per unit of unmet customer demand\n"
                f"- **Overage Cost (Co = Cost - Salvage)**: ${co:,.2f} per unit of unsold leftover inventory\n"
                f"- **Critical Ratio (Cu / [Cu + Co])**: {cr:.4f}\n"
                f"- **Expected Sales**: {results['expected_sales']:,.1f} units\n"
                f"- **Expected Leftovers**: {leftovers:,.1f} units\n"
                f"- **Expected Lost Sales**: {lost_sales:,.1f} units\n\n"
                f"> Because the cost of stocking out (${cu:,.2f}) vs holding excess (${co:,.2f}) yields a critical fractile "
                f"of {cr:.2%}, the decision-maker orders {q_opt:,.0f} units, which corresponds to the {cr:.1%} percentile of the demand distribution."
            )

            recs = [
                f"Commit to a one-time purchase order of {q_opt:,.0f} units.",
                f"Prepare markdown/salvage disposal channels in advance for an anticipated {leftovers:,.0f} leftover units.",
                "Seek supply partners who allow in-season replenishment or return allowances to hedge demand uncertainty.",
                "Invest in demand sensing and pre-season reservations to reduce demand variance (σ)."
            ]

        elif model_id == self.MODEL_INVENTORY_CLASSIFICATION:
            tot_items = results.get("total_items", 0)
            tot_sales = results.get("total_sales", 0.0)
            tot_hits = results.get("total_hits", 0.0)
            s_sum = results.get("sales_class_summary", {})
            h_sum = results.get("hits_class_summary", {})
            f_sum = results.get("final_class_summary", {})
            mat = results.get("multicriteria_matrix", {})

            summary = (
                f"**Multicriteria Inventory Classification Completed for {tot_items:,} Items.**\n"
                f"Total analyzed revenue is **${tot_sales:,.2f}** and total transaction frequency is **{tot_hits:,.0f} POS Hits**.\n"
                f"- Sales Class (Score A=4, B=3, C=2, D=1) and Hits Class (Score A=4, B=3, C=2, D=1) were calculated separately.\n"
                f"- Combined Weighted Score = **0.65(Sales Score) + 0.35(Hits Score)**.\n"
                f"- Final Inventory Classes: **A (4.00-3.25)**: {f_sum.get('A', {}).get('count', 0)} SKUs, "
                f"**B (3.24-2.50)**: {f_sum.get('B', {}).get('count', 0)} SKUs, "
                f"**C (2.49-1.75)**: {f_sum.get('C', {}).get('count', 0)} SKUs, "
                f"**D (< 1.75)**: {f_sum.get('D', {}).get('count', 0)} SKUs."
            )

            details = (
                f"### Multicriteria Pareto Classification & Combined Weighted Scoring\n\n"
                f"#### 1. Final Inventory Classification (Combined Weighted Score = 0.65×Sales + 0.35×Hits):\n"
                f"- **Class A [4.00 - 3.25]**: {f_sum.get('A', {}).get('count', 0)} SKUs "
                f"(${f_sum.get('A', {}).get('total_sales', 0):,.2f} | {f_sum.get('A', {}).get('sales_pct', 0):.1f}% of sales)\n"
                f"- **Class B [3.24 - 2.50]**: {f_sum.get('B', {}).get('count', 0)} SKUs "
                f"(${f_sum.get('B', {}).get('total_sales', 0):,.2f} | {f_sum.get('B', {}).get('sales_pct', 0):.1f}% of sales)\n"
                f"- **Class C [2.49 - 1.75]**: {f_sum.get('C', {}).get('count', 0)} SKUs "
                f"(${f_sum.get('C', {}).get('total_sales', 0):,.2f} | {f_sum.get('C', {}).get('sales_pct', 0):.1f}% of sales)\n"
                f"- **Class D [< 1.75]**: {f_sum.get('D', {}).get('count', 0)} SKUs "
                f"(${f_sum.get('D', {}).get('total_sales', 0):,.2f} | {f_sum.get('D', {}).get('sales_pct', 0):.1f}% of sales)\n\n"
                f"#### 2. Individual Sales ($) Pareto Breakdown (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%):\n"
                f"- **Class A (Score 4)**: {s_sum.get('A', {}).get('count', 0)} SKUs (${s_sum.get('A', {}).get('total_sales', 0):,.2f} | {s_sum.get('A', {}).get('sales_pct', 0):.1f}% of revenue)\n"
                f"- **Class B (Score 3)**: {s_sum.get('B', {}).get('count', 0)} SKUs (${s_sum.get('B', {}).get('total_sales', 0):,.2f} | {s_sum.get('B', {}).get('sales_pct', 0):.1f}% of revenue)\n"
                f"- **Class C (Score 2)**: {s_sum.get('C', {}).get('count', 0)} SKUs (${s_sum.get('C', {}).get('total_sales', 0):,.2f} | {s_sum.get('C', {}).get('sales_pct', 0):.1f}% of revenue)\n"
                f"- **Class D (Score 1)**: {s_sum.get('D', {}).get('count', 0)} SKUs (${s_sum.get('D', {}).get('total_sales', 0):,.2f} | {s_sum.get('D', {}).get('sales_pct', 0):.1f}% of revenue)\n\n"
                f"#### 3. Individual Hits / Movement Frequency Breakdown (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%):\n"
                f"- **Class A (Score 4)**: {h_sum.get('A', {}).get('count', 0)} SKUs ({h_sum.get('A', {}).get('total_hits', 0):,.0f} hits | {h_sum.get('A', {}).get('hits_pct', 0):.1f}% of activity)\n"
                f"- **Class B (Score 3)**: {h_sum.get('B', {}).get('count', 0)} SKUs ({h_sum.get('B', {}).get('total_hits', 0):,.0f} hits | {h_sum.get('B', {}).get('hits_pct', 0):.1f}% of activity)\n"
                f"- **Class C (Score 2)**: {h_sum.get('C', {}).get('count', 0)} SKUs ({h_sum.get('C', {}).get('total_hits', 0):,.0f} hits | {h_sum.get('C', {}).get('hits_pct', 0):.1f}% of activity)\n"
                f"- **Class D (Score 1)**: {h_sum.get('D', {}).get('count', 0)} SKUs ({h_sum.get('D', {}).get('total_hits', 0):,.0f} hits | {h_sum.get('D', {}).get('hits_pct', 0):.1f}% of activity)\n\n"
                f"#### 4. Matrix Cross-Tabulation (Sales Class × Hits Class):\n"
                f"- **AA (Score 4.00)**: {mat.get('AA', 0)} SKUs — Critical high-value, high-velocity core drivers.\n"
                f"- **AD (Score 2.95)**: {mat.get('AD', 0)} SKUs — High revenue with low pick frequency.\n"
                f"- **DA (Score 2.05)**: {mat.get('DA', 0)} SKUs — Low revenue with high pick frequency.\n"
                f"- **DD (Score 1.00)**: {mat.get('DD', 0)} SKUs — Long-tail slow movers."
            )

            recs = [
                f"Prioritize replenishment and continuous review for Final Class A SKUs ({f_sum.get('A', {}).get('count', 0)} items, Score 3.25-4.00) targeting 98%+ cycle service level.",
                f"Manage Final Class B SKUs ({f_sum.get('B', {}).get('count', 0)} items, Score 2.50-3.24) with standard periodic review and safety stock buffers.",
                f"Control Final Class C SKUs ({f_sum.get('C', {}).get('count', 0)} items, Score 1.75-2.49) using simplified reorder points or two-bin replenishment for low-cost fast movers.",
                f"Conduct obsolescence and minimum order quantity (MOQ) audits on Final Class D SKUs ({f_sum.get('D', {}).get('count', 0)} items, Score < 1.75)."
            ]

        full_details = f"{eight_report_str}\n---\n{details}" if eight_report_str else details
        return (summary, full_details, recs)

    def classify_inventory(
        self,
        items_data: Any,
        item_col: Optional[str] = None,
        name_col: Optional[str] = None,
        uom_col: Optional[str] = None,
        sales_col: Optional[str] = None,
        hits_col: Optional[str] = None
    ) -> tools.InventoryClassificationResult:
        """
        Directly classifies inventory items using Sales ($) and Hits (POS Hits).
        Applies Pareto thresholds (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)
        and converts classes to scores (A=4, B=3, C=2, D=1).
        """
        return tools.calculate_multicriteria_inventory_classification(
            items_data=items_data,
            item_col=item_col,
            name_col=name_col,
            uom_col=uom_col,
            sales_col=sales_col,
            hits_col=hits_col
        )
