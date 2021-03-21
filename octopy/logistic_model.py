from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


class LogisticModel:
    def fit(
        self, home_team_name, away_team_name, home_score, away_score, team_names=None
    ):
        """
        Fit a logistic regression model with three classes.

        Parameters
        ----------
        home_team_name: list
            A list of home team name.

        away_team_name: list
            A list of away team name.

        home_score: list
            A list of home team score

        away_score: list
            A list of away team score

        team_names: list, optional
            A list of team names that contains all teams. Useful when new teams are not
            in home_team_name and away_team_name history. When None it default to all teams
            included in home_team_name and away_team_name.

        Attributes
        ----------

        team_encoding_: OneHotEncoder
            A sklearn OneHotEncoder class that contains the team encoding.

        model_: LogisticRegression
            A sklearn LogisticRegression model fitted to the data.

        """
        home_team_name, away_team_name, home_score, away_score = [
            np.array(x)
            for x in [home_team_name, away_team_name, home_score, away_score]
        ]
        if team_names is None:
            team_names = np.array(list(home_team_name) + list(away_team_name)).reshape(
                -1, 1
            )
        else:
            team_names = np.array(team_names).reshape(-1, 1)

        self.team_encoding_ = OneHotEncoder(sparse=False).fit(team_names)

        home_dummies = self.team_encoding_.transform(home_team_name.reshape(-1, 1))
        away_dummies = self.team_encoding_.transform(away_team_name.reshape(-1, 1))

        X = np.concatenate([home_dummies, away_dummies], 1)
        y = np.sign(home_score - away_score)

        model = LogisticRegression(
            penalty="l2", fit_intercept=False, multi_class="ovr", C=1
        )
        model.fit(X, y)
        self.model_ = model

    def get_coef(self):
        """

        Get the coefficients of three logistic models for each team.

        """
        home_feature_names = [
            x.replace("x0", "home") for x in self.team_encoding_.get_feature_names()
        ]
        away_feature_names = [
            x.replace("x0", "away") for x in self.team_encoding_.get_feature_names()
        ]
        coeffs = pd.DataFrame(
            self.model_.coef_,
            index=self.model_.classes_,
            columns=[home_feature_names + away_feature_names],
        ).T
        return coeffs.rename(columns={-1: "away wins", 0: "draw", 1: "home wins"})

    def check_teams(self, home_team_name, away_team_name):
        """Check if team are encoded."""
        assert (
            home_team_name in self.team_encoding_.categories_[0]
        ), f"{home_team_name} is recognized. It was not in the training data."
        assert (
            away_team_name in self.team_encoding_.categories_[0]
        ), f"{away_team_name} is recognized. It was not in the training data."

    def predict_winner(self, home_team_name, away_team_name):
        """
        Predict the winner.

        Parameters
        ----------
        home_team_name: str
            Home team name.

        away_team_name: str
            Away team name.

        Returns
        -------
        The name of the winning team or "draw".

        """

        check_is_fitted(self.model_)
        self.check_teams(home_team_name, away_team_name)
        home_dummies = self.team_encoding_.transform(
            np.array(home_team_name).reshape(-1, 1)
        )
        away_dummies = self.team_encoding_.transform(
            np.array(away_team_name).reshape(-1, 1)
        )
        X = np.concatenate([home_dummies, away_dummies], 1)
        pred = self.model_.predict(X)
        if pred == 0:
            return "draw"
        if pred > 0:
            return str(home_team_name)
        else:
            return str(away_team_name)

    def predict_proba(self, home_team_name, away_team_name):
        """
        Predict the probabilities of draw and win for each team..

        Parameters
        ----------
        home_team_name: str
            Home team name.

        away_team_name: str
            Away team name.

        Returns
        -------
        A dataframe with the probabilities.

        """
        check_is_fitted(self.model_)
        self.check_teams(home_team_name, away_team_name)
        home_team_name = np.array(home_team_name)
        away_team_name = np.array(away_team_name)
        home_dummies = self.team_encoding_.transform(home_team_name.reshape(-1, 1))
        away_dummies = self.team_encoding_.transform(away_team_name.reshape(-1, 1))
        X = np.concatenate([home_dummies, away_dummies], 1)
        return pd.DataFrame(
            self.model_.predict_proba(X),
            index=["probability"],
            columns=self.model_.classes_,
        ).rename(columns={-1: f"{away_team_name}", 0: "draw", 1: f"{home_team_name}"})
