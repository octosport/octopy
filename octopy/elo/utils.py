import jax.numpy as jnp
from jax import jit


@jit
def get_log_loss(scoreA, scoreB, pA, pB, pD=0):
    """return the log loss given the score and probabilities."""

    return (
        (scoreA > scoreB) * jnp.log(pA)
        + (scoreA == scoreB) * jnp.log(pD)
        + (scoreA < scoreB) * jnp.log(pB)
    )


@jit
def get_winner(A, B):
    """Return an interger that represents who won the match: 0 for A, 1 for D and 2 for B"""
    return (A == B) * 1.0 + (A < B) * 2.0
