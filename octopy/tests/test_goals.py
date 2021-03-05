import numpy as np
import pytest
from ..goals import PoissonDistribution


def test_proba():
    poisson_goals = PoissonDistribution(1.85, 0.98)
    np.testing.assert_almost_equal(
        poisson_goals.predict_proba(1, 0), 0.1091737792884785, 5
    )
    np.testing.assert_almost_equal(
        poisson_goals.predict_proba(2, 3), 0.0158411626833959, 5
    )


def test_positive_goal_exception():
    poisson_goals = PoissonDistribution(1.85, 0.98)
    with pytest.raises(Exception):
        poisson_goals.predict_proba(-1, 0)
        poisson_goals.predict_proba(1, -2)


def test_integer_goal_exception():
    poisson_goals = PoissonDistribution(1.85, 0.98)
    with pytest.raises(Exception):
        poisson_goals.predict_proba(1.2, 0)
        poisson_goals.predict_proba(0, 3.2)
