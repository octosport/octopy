import jax.numpy as jnp
from jax import jit
from jax import nn

__EPS__ = 1e-12


@jit
def get_log_loss(score1, score2, p1, p2, pt=0):
    return (
        (score1 > score2) * jnp.log(p1)
        + (score1 == score2) * jnp.log(pt)
        + (score1 < score2) * jnp.log(p2)
    )


@jit
def predict_proba(params, home_rating, away_rating, has_tie):
    dr = (home_rating - away_rating) * params["beta"]
    gamma = nn.relu(params["gamma"]) * has_tie
    pA = jnp.clip(nn.sigmoid(dr - gamma), __EPS__, 1 - __EPS__)
    pB = jnp.clip(nn.sigmoid(-dr - gamma), __EPS__, 1 - __EPS__)
    pD = nn.relu(1.0 - pA - pB) * has_tie
    s = pA + pB + pD
    return pA / s, pD / s, pB / s


@jit
def get_winner(a, b):
    return (a == b) * 1.0 + (a < b) * 2.0
