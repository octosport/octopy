from sklearn.preprocessing import LabelEncoder
import numpy as np


class EloDataset:
    def __init__(
        self,
        valid_fration=0.2,
        test_fraction=0.2,
        time=None,
        valid_date=None,
        test_date=None,
    ):
        """
        Dataset object to train the Elo rating model. The data are split in three sets:
        the train set where the model is fitted, the validation set used for early stopping and the test set.
        Sets are split using the fraction of the total dataset size if no time and dates are provided.

        Parameters
        ----------

        test_fraction: double
            Define the size of the test data in % of the dataset total size.

        valid_fration: double
            Define the size of the validation data in % of the dataset total size.

        time: array of datetime (optional)
            Time index for matches.

        valid_date: date (optional)
            Date to split between train (before it) and validation set.

        test_date: date (optional)
            Date to split between validation (before it) and test set.

        """

        self.time = time
        self.test_date = test_date
        self.valid_date = valid_date
        self.test_fraction = test_fraction
        self.valid_fration = valid_fration

    def encode_teams(self, team_names):
        """Encode team name."""
        self.le_ = LabelEncoder().fit(list(team_names[:, 0]) + list(team_names[:, 1]))
        team_index = np.zeros_like(team_names)
        team_index[:, 0] = self.le_.transform(team_names[:, 0])
        team_index[:, 1] = self.le_.transform(team_names[:, 1])
        self.n_teams_ = len(self.le_.classes_)
        self.team_index_ = np.array(team_index).astype(np.int32)

    def split_train_test(self, team_names, scores):
        """
        Split the data for training.

        Parameters
        ----------
        team_names: string array of size (n_matches,2)
            Array of team names with names of team A in column 0 and names of team B in column 1.

        scores: array of size (n_matches,2)
            Array of goals score by Team A (column 0) and team B (column 1)

        Returns
        -------

        """
        self.scores_ = np.array(scores).astype(np.float32)
        self.encode_teams(team_names)

        if self.time is None:
            time = np.array(range(len(scores)))
        else:
            time = self.time

        if None not in (self.test_date, self.valid_date):
            assert time is not None
            test_date = self.test_date
            valid_date = self.valid_date
            split_type_ = "date"
        else:
            test_idx = int((1 - self.test_fraction) * len(time))
            train_idx = int((1 - self.test_fraction - self.valid_fration) * len(time))
            test_date = time[test_idx]
            valid_date = time[train_idx]
            split_type_ = "fraction"

        self.train_index_ = time < valid_date
        self.valid_index_ = (time > valid_date) & (time < test_date)
        self.test_index_ = time > test_date
        print(
            f"dataset split using {split_type_}: train size {int(sum(self.train_index_))}, validation size {int(sum(self.valid_index_))}, test size {int(sum(self.test_index_))}"
        )

    def get_train_split(self):
        return {
            "team_index": self.team_index_[self.train_index_, :],
            "scores": self.scores_[self.train_index_, :],
        }

    def get_dataset(self):
        assert hasattr(
            self, "scores_"
        ), "split_train_test(team_names, scores) needs to be call first."
        return {"team_index": self.team_index_, "scores": self.scores_}
