from scipy.stats import poisson
from .utils import assert_positive_int, assert_positive


class PoissonDistribution:
    def __init__(self, home_lambda, away_lambda):
        """
        Soccer result probability prediction using the Poisson distribution with independence hypothesis.

        Parameters
        ----------
        home_lambda : float
            Number of goals the home team is expected to score (home strength).

        away_lambda : float
            Number of goals the away team is expected to score (away strength).

        """
        assert_positive(home_lambda, "home_lambda should be a positive or null.")
        assert_positive(away_lambda, "away_lambda should be a positive or null.")
        self.home_lambda = home_lambda
        self.away_lambda = away_lambda

    def predict_proba(self, home_score, away_score):
        """
        Predict the probability of a given results.

        Parameters
        ----------
        home_score : int
            Home team result.

        away_score : int
            Away team result.

        Returns
        -------
        The probability of the result "home_score - away_score".

        """
        assert_positive_int(
            home_score, "home_score should be a positive or null integer."
        )
        assert_positive_int(
            away_score, "away_score should be a positive or null integer."
        )
        k_home = home_score
        k_away = away_score

        return poisson.pmf(k_home, self.home_lambda) * poisson.pmf(
            k_away, self.away_lambda
        )
