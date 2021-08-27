from jax import nn
from jax import jit, grad
import jax.numpy as jnp
from tqdm import notebook
import jax.ops as jop
from jax import lax
import numpy as np
from ..elo.utils import predict_proba, get_log_loss, get_winner


class EloRatingNet:
    def __init__(self, n_teams, tie=True):
        self.tie = tie
        self.n_teams = n_teams

    def init_params(self):
        return dict(
            beta=1.0,
            gamma=0.2,
            lr=0.19,
            ha=jnp.array([0.0 for k in range(self.n_teams)]),
            init=jnp.array([100.0 for k in range(self.n_teams)])
        )

    def get_train_function(self, keep_rating=True):

        @jit
        def update_ratings(
                params, home_rating, away_rating, home, away, winner, rating
        ):

            p_home, _, p_away = predict_proba(params, home_rating, away_rating,self.tie)

            operand = nn.relu(params["lr"])
            delta_home_d = lax.cond(
                winner == 1.0,
                lambda x: x * (p_away - p_home),
                lambda x: 0.0,
                operand,
            )
            delta_away_d = -delta_home_d

            delta_home_h = lax.cond(
                winner == 0.0,
                lambda x: x * (1 - p_home),
                lambda x: 0.0,
                operand
            )
            delta_away_h = lax.cond(
                winner == 0.0,
                lambda x: x * (0 - p_away),
                lambda x: 0.0,
                operand
            )

            delta_home_a = lax.cond(
                winner == 2.0,
                lambda x: x * (0 - p_home),
                lambda x: 0.0,
                operand
            )
            delta_away_a = lax.cond(
                winner == 2.0,
                lambda x: x * (1 - p_away),
                lambda x: 0.0,
                operand
            )

            delta_home = delta_home_d + delta_home_h + delta_home_a
            delta_away = delta_away_d + delta_away_h + delta_away_a

            rating = jop.index_add(rating, home, delta_home)
            rating = jop.index_add(rating, away, delta_away)
            return rating

        def inner_loop(carry, dataset_obs, keep_rating=keep_rating):
            rating = carry["rating"]
            params = carry["params"]

            home, away = dataset_obs["team_index"][0], dataset_obs["team_index"][1]
            score1, score2 = dataset_obs["scores"][0], dataset_obs["scores"][1]

            p1, pt, p2 = predict_proba(params, rating[home], rating[away],self.tie)
            loss = get_log_loss(score1, score2, p1, p2, pt)

            winner = get_winner(score1, score2)
            carry["rating"] = update_ratings(
                params, rating[home], rating[away], home, away, winner, rating
            )

            if keep_rating:
                return carry, [loss, rating]
            else:
                return carry, [loss, jnp.nan]

        def scan_loss(params, dataset):
            init = params["init"]
            carry = dict()
            carry["params"] = params
            carry["rating"] = init
            carry, output = lax.scan(inner_loop, carry, dataset)

            return {
                "carry": carry,
                "loss_history": output[0],
                "rating": output[1],
            }

        def train_loss(params, dataset):
            output = scan_loss(params, dataset)
            return -jnp.mean(output["loss_history"])

        return train_loss, scan_loss

    def get_split_losses(self, loss_history, dataset):
        loss_history = np.array(loss_history)
        train_loss = -np.mean(loss_history[dataset.train_index_])
        valid_loss = -np.mean(loss_history[dataset.valid_index_])
        test_loss = -np.mean(loss_history[dataset.test_index_])
        return train_loss, valid_loss, test_loss

    an_loss(best_p
    def optimise(
            self,
            dataset,
            learning_rate=0.1,
            max_step=10000,
            early_stopping=100,
            verbose=50,
    ):
        '''
        perform gradient descent optimization.

        Parameters
        ----------
        dataset: EloDataset
            The EloDataset object that contains the data.

        learning_rate: double
            Gradient descent learning rate.

        max_step: int
            Maximum number of gradient step.

        early_stopping: int
            Stop optimizing if the validation loss has not decreased since the last number of step specified
            by this parameter.

        verbose: int
            Print losses ever nth steps.

        '''



        train_loss, scan_loss = self.get_train_function(keep_rating=False)

        jit_grad = jit(grad(train_loss))
        jit_scan_loss = jit(scan_loss)
        params = self.init_params()
        loss_path = []
        min_loss = 1e5
        train_data = dataset.get_train_split()
        full_data = dataset.get_dataset()
        for i in notebook.tqdm(range(max_step)):
            grads = jit_grad(params, train_data)
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

            output = jit_scan_loss(params, full_data)
            loss_history = output["loss_history"]
            train_loss, valid_loss, test_loss = self.get_split_losses(loss_history, dataset)

            #stopping rules
            if min_loss > valid_loss:
                min_loss = valid_loss
                if i % verbose == 0:
                    print(f'train_loss: {train_loss}, valid_loss: {valid_loss}, test_loss: {test_loss}')
                stopping = 0
                best_params = jnp.copy(params)
            else:
                stopping += 1

            if stopping > early_stopping:
                print("stop learning optimal it: {}".format(i))
                break

            loss_path = loss_path + [jnp.array([train_loss, valid_loss, test_loss])]

        output = jit_scan_loss(best_params, full_data)
        loss_history = output["loss_history"]
        self.params_ = best_params
        self.loss_path_ = loss_path
        self.loss_history_ = loss_history
        self.output_ = output



