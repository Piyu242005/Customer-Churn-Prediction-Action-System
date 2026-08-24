from src.retention import retention_action, risk_band, roi_estimate


def test_risk_band():
    assert risk_band(10) == "Low"
    assert risk_band(50) == "Medium"
    assert risk_band(90) == "High"


def test_retention_action():
    assert "pricing" in retention_action("Monthly Charges", "High").lower()
    assert "contract" in retention_action("Contract", "High").lower()


def test_roi_estimate():
    result = roi_estimate(100, 10000, 20, 50)
    assert result["expected_saved"] == 5000
    assert result["intervention_cost"] == 2000
    assert result["net_value"] == 3000
    assert result["roi_percent"] == 150
