import numpy as np
import pytest

from churn.config import EconomicsConfig
from churn.economics import (
    baselines,
    best_threshold,
    campaign_profit,
    customer_value,
    expected_value,
)

ECON = EconomicsConfig(horizon_months=10, retention_success_rate=0.5, cost_per_contact=20.0)


def test_customer_value():
    assert customer_value(np.array([10.0, 50.0]), ECON).tolist() == [100.0, 500.0]


def test_campaign_profit_by_hand():
    y = np.array([1, 0, 1])
    contact = np.array([True, True, False])
    value = np.array([100.0, 100.0, 100.0])
    # Un desertor contactado: 0.5 * 100 = 50; dos contactos: 40.
    assert campaign_profit(y, contact, value, ECON) == pytest.approx(10.0)


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    value = np.full(4, 1000.0)
    result = best_threshold(y, proba, value, ECON)
    assert 0.2 < result.threshold <= 0.8
    assert result.churners_contacted == 2
    assert result.profit == pytest.approx(baselines(y, value, ECON)["perfect_oracle"])


def test_baselines_order():
    y = np.array([0, 1, 0, 0])
    value = np.full(4, 100.0)
    b = baselines(y, value, ECON)
    assert b["perfect_oracle"] >= b["no_campaign"]
    assert b["contact_everyone"] == pytest.approx(0.5 * 100 - 4 * 20)


def test_expected_value_sign():
    ev = expected_value(np.array([0.0, 1.0]), np.array([100.0, 100.0]), ECON)
    assert ev[0] == pytest.approx(-20.0)
    assert ev[1] == pytest.approx(30.0)
