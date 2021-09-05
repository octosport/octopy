from jax import nn
from jax import jit, grad
import jax.numpy as jnp
from tqdm import notebook
import jax.ops as jop
from jax import lax
import numpy as np
from ..elo.utils import predict_proba, get_log_loss, get_winner


class EloRatingNet:
    '''
    Train Elo rating model using recurrent neural network.

    Parameters
    ----------
    n_teams: int
        Number of team to rate

    has_tie: bool
        If True the probability of a draw if also computed

    '''

    def __init__(self, n_teams, has_tie=True):
        self.has_tie = has_tie
        self.n_teams = n_teams

    def init_params(self):
        '''Set of model initial parameters.'''
        return dict(
            beta=1.0,
            gamma=0.2,
            lr=0.1,
            init=jnp.array([1000.0 for k in range(self.n_teams)]),
        )

    def get_train_function(self, keep_rating=True):
        @jit
        def update_ratings(
                params, teamA_rating, teamB_rating, teamA_idx, teamB_idx, winner, rating
        ):
            '''Update rating step'''

            pA, _, pB = predict_proba(
                params, teamA_rating, teamB_rating, self.has_tie
            )

            operand = nn.relu(params["lr"])
            delta_A_d = lax.cond(
                winner == 1.0,
                lambda x: x * (pB - pA),
                lambda x: 0.0,
                operand,
            )
            delta_B_d = -delta_A_d

            delta_A_win = lax.cond(
                winner == 0.0, lambda x: x * (1 - pA), lambda x: 0.0, operand
            )
            delta_B_lose = lax.cond(
                winner == 0.0, lambda x: x * (0 - pB), lambda x: 0.0, operand
            )

            delta_A_lose = lax.cond(
                winner == 2.0, lambda x: x * (0 - pA), lambda x: 0.0, operand
            )
            delta_B_win = lax.cond(
                winner == 2.0, lambda x: x * (1 - pB), lambda x: 0.0, operand
            )

            delta_A = delta_A_d + delta_A_win + delta_A_lose
            delta_B = delta_B_d + delta_B_lose + delta_B_win

            rating = jop.index_add(rating, teamA_idx, jnp.tanh(delta_A))
            rating = jop.index_add(rating, teamB_idx, jnp.tanh(delta_B))
            return rating

        def scan_function(carry, dataset, keep_rating=keep_rating):
            '''Predict and rate for each data in the dataset.'''

            rating = carry["rating"]
            params = carry["params"]

            teamA_idx, teamB_idx = dataset["team_index"][0], dataset["team_index"][1]
            score1, score2 = dataset["scores"][0], dataset["scores"][1]

            p1, pt, p2 = predict_proba(params, rating[teamA_idx], rating[teamB_idx], self.has_tie)
            loss = get_log_loss(score1, score2, p1, p2, pt)

            winner = get_winner(score1, score2)
            carry["rating"] = update_ratings(
                params, rating[teamA_idx], rating[teamB_idx], teamA_idx, teamB_idx, winner, rating
            )

            if keep_rating:
                return carry, [loss, rating]
            else:
                return carry, [loss, jnp.nan]

        def scan_loss(params, dataset):
            '''Predict and rate the entire dataset given the paramters.'''
            init = params["init"]
            carry = dict()
            carry["params"] = params
            carry["rating"] = init
            carry, output = lax.scan(scan_function, carry, dataset)

            return {
                "carry": carry,
                "loss_history": output[0],
                "rating": output[1],
            }

        def negative_average_log_loss(params, dataset):
            output = scan_loss(params, dataset)

            return -jnp.mean(output["loss_history"])

        return negative_average_log_loss, scan_loss

    def get_split_losses(self, loss_history, dataset, round=None):
        loss_history = np.array(loss_history)
        train_loss = -np.mean(loss_history[dataset.train_index_])
        valid_loss = -np.mean(loss_history[dataset.valid_index_])
        test_loss = -np.mean(loss_history[dataset.test_index_])
        if round is None:
            return train_loss, valid_loss, test_loss
        else:
            return [np.round(x, round) for x in [train_loss, valid_loss, test_loss]]

    def optimise(
        self,
        dataset,
        learning_rate=0.1,
        max_step=10000,
        early_stopping=100,
        verbose=50,
    ):
        """
        Perform gradient descent optimization.

        Parameters
        ----------
        dataset: EloDataset
            The EloDataset object that contains the data.

        learning_rate: double
            Gradient descent learning rate.

        max_step: int
            Maximum number of gradient steps.

        early_stopping: int
            Stop optimizing if the validation loss has not decreased since the last number of step specified
            by this parameter.

        verbose: int
            Print losses ever nth steps.

        """

        neg_average_log_loss_fn, scan_loss_fn = self.get_train_function(keep_rating=True)

        jit_nll_grad_fn = jit(grad(neg_average_log_loss_fn))
        jit_scan_loss_fn = jit(scan_loss_fn)
        params = self.init_params()
        loss_path = []
        min_loss = 1e5
        train_data = dataset.get_train_split()
        full_data = dataset.get_dataset()
        for i in notebook.tqdm(range(max_step)):
            grads = jit_nll_grad_fn(params, train_data)
            for key, val in params.items():
                if isinstance(params[key], list):
                    params[key] = jnp.array(
                        [
                            v - learning_rate * grads[key][k]
                            for k, v in enumerate(params[key])
                        ]
                    )
                else:
                    params[key] = val - learning_rate * grads[key]

            output = jit_scan_loss_fn(params, full_data)
            loss_history = output["loss_history"]
            train_loss, valid_loss, test_loss = self.get_split_losses(
                loss_history, dataset
            )

            # stopping rules
            if min_loss > valid_loss:
                min_loss = valid_loss
                if i % verbose == 0:
                    print(
                        f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, test_loss: {test_loss:.4f}"
                    )
                stopping = 0
                best_params = params.copy()
            else:
                stopping += 1

            if stopping > early_stopping:
                n_iter = i - stopping
                print("optimal stopping at iter: {}".format(i))
                break

            n_iter = i
            best_params = params.copy()
            loss_path = loss_path + [jnp.array([train_loss, valid_loss, test_loss])]

        output = jit_scan_loss_fn(best_params, full_data)
        loss_history = output["loss_history"]
        best_params["n_iter"] = n_iter
        self.best_params_ = best_params
        self.loss_path_ = loss_path
        self.loss_history_ = loss_history
        self.output_ = output

        self.ratings_ = dict(zip(dataset.le_.classes_, output["carry"]["rating"]))

    def predict_proba(self, teamA, teamB):
        '''
        Predict the probability of wining for each team. If self.tie=True, the probability of draw is added.

        Parameters
        ----------
        teamA: string
            Name of teamA

        teamB: string
            Name of teamB

        Returns
        -------

        A dict containing the probabilities.

        '''
        teamA_rating = self.ratings_[teamA]
        teamB_rating = self.ratings_[teamB]
        pA, pD, pB = predict_proba(
            self.best_params_, teamA_rating=teamA_rating, teamB_rating=teamB_rating,has_tie=self.has_tie
        )
        return {f"{teamA}": pA, "Draw": pD, f"{teamB}": pB}
