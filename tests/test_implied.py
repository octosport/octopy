import numpy as np
from octopy.octopy.implied import ImpliedProbability


def test_shin_method():
    implied_model = ImpliedProbability(method="shin")
    implied_model.convert(4.20, 3.70, 1.95)

    implied_probabilities = implied_model.implied_probabilities
    np.testing.assert_almost_equal(
        implied_probabilities, np.array([0.23157, 0.26357, 0.50485]), 5
    )
