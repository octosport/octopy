# Load Data
import numpy as np
from scipy.optimize import bisect


def _get_shin_implied_probabilities(z, pi):
    normalization = sum(pi)
    return ((z ** 2 + 4 * (1 - z) * pi ** 2 / normalization) ** 0.5 - z) / (2 - 2 * z)


def _get_shin_normalization(z, pi):
    implied_probabilities = _get_shin_implied_probabilities(z, pi)
    return 1 - sum(implied_probabilities)


def _get_power_normalization(k, pi):
    implied_probabilities = pi ** k
    return 1 - sum(implied_probabilities)


class ImpliedProbability:
    @property
    def implied_probabilities(self):
        return self._implied_probabilities

    @property
    def margins(self):
        return self._margins

    def __init__(self, method="shin"):
        """
        Convert bookmaker odds into probabilities.

        Parameters
        ----------
        method : str
            Method used to find the implied probabilities. Available methods are:
                - basic
                - shin
                - power
        """
        assert method in [
            "power",
            "shin",
            "additive",
            "multiplicative",
        ], "Method {} is not available. Please choose a valid method.".format(method)
        self.method = method
        self._implied_probabilities = None
        self._margins = None

    def convert(self, home_odd, draw_odd, away_odd):
        """
        Convert the bookmakers odds into probabilities.

        Parameters
        ----------
        home_odd : float
            Home bookmaker odd.

        draw_odd : float
            draw bookmaker odd

        away_odd : float
            Away bookmaker odd

        Returns
        -------
        self

        """
        try:
            odds = np.array([home_odd, draw_odd, away_odd])
            pi = 1 / odds

            if self.method == "multiplicative":
                normalization = sum(pi)
                implied_probabilities = pi / normalization
            if self.method == "shin":
                z_opt = bisect(_get_shin_normalization, 0, 10, args=(pi))
                implied_probabilities = _get_shin_implied_probabilities(z_opt, pi)
            if self.method == "power":
                k_opt = bisect(_get_power_normalization, 0, 100, args=(pi))
                implied_probabilities = pi ** k_opt
            if self.method == "additive":
                implied_probabilities = pi + 1 / len(odds) * (1 - sum(pi))
            self._implied_probabilities = implied_probabilities
            self._margins = pi - self.implied_probabilities
        except:
            self._implied_probabilities = [np.nan] * len(odds)
            self._margins = [np.nan] * len(odds)
        return self
