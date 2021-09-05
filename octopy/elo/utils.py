import jax.numpy as jnp
from jax import jit
from jax import nn

__EPS__ = 1e-12

@jit
def predict_proba(params, home_rating, away_rating, has_tie):
    dr = (home_rating - away_rating) * params["beta"]
    gamma = nn.relu(params["gamma"]) * has_tie
    pA = jnp.clip(nn.sigmoid(dr - gamma), __EPS__, 1 - __EPS__)
    pB = jnp.clip(nn.sigmoid(-dr - gamma), __EPS__, 1 - __EPS__)
    pD = nn.relu(1.0 - pA - pB) * has_tie
    s = pA + pB + pD
    return [float(x) for x in [pA / s, pD / s, pB / s]]

@jit
def get_log_loss(scoreA, scoreB, pA, pB, pD=0):
    '''return the log loss given the score and probabilities.'''

    return (
        (scoreA > scoreB) * jnp.log(pA)
        + (scoreA == scoreB) * jnp.log(pD)
        + (scoreA < scoreB) * jnp.log(pB)
    )


@jit
def get_winner(A, B):
    '''Return an interger that represents who won the match: 0 for A, 1 for D and 2 for B'''
    return (A == B) * 1.0 + (A < B) * 2.0
