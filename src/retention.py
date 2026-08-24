"""Business retention rules shared by the dashboard and tests."""

from __future__ import annotations


def retention_action(top_driver: str, risk_level: str) -> str:
    driver = str(top_driver).lower()
    risk = str(risk_level).lower()

    if risk == "low":
        return "Monitor engagement; no immediate intervention required."
    if "discount" in driver or "price" in driver or "charges" in driver:
        return "Review pricing and offer a targeted retention incentive."
    if "contract" in driver:
        return "Offer a longer-term contract with a loyalty benefit."
    if "support" in driver or "tech" in driver:
        return "Trigger proactive support outreach and service-health review."
    if "quantity" in driver or "engagement" in driver:
        return "Launch a targeted engagement campaign and product education."
    if "profit" in driver or "revenue" in driver:
        return "Prioritize account review and tailor a value-preserving offer."
    return "Assign the customer to retention review and proactive outreach."


def risk_band(probability: float, low: float = 30.0, high: float = 70.0) -> str:
    if probability < low:
        return "Low"
    if probability < high:
        return "Medium"
    return "High"


def roi_estimate(
    customers_targeted: int,
    revenue_at_risk: float,
    intervention_cost_per_customer: float,
    expected_save_rate: float,
) -> dict[str, float]:
    customers = max(int(customers_targeted), 0)
    revenue = max(float(revenue_at_risk), 0.0)
    save_rate = min(max(float(expected_save_rate), 0.0), 100.0) / 100.0
    cost = customers * max(float(intervention_cost_per_customer), 0.0)
    expected_saved = revenue * save_rate
    net_value = expected_saved - cost
    roi = (net_value / cost * 100.0) if cost else 0.0
    return {
        "expected_saved": expected_saved,
        "intervention_cost": cost,
        "net_value": net_value,
        "roi_percent": roi,
    }
