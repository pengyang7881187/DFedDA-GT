import jax.nn
import jax.numpy as jnp

from MyTyping import *


def L2_loss(y, y_hat):
    return 0.5 * jnp.square(y - y_hat)


def cross_entropy_loss(y, prob_hat):
    return -(y * jnp.log(prob_hat) + (1. - y) * jnp.log(1.-prob_hat))


# Linear regression with L2 loss, xi[0] = y, xi[1] = 1, xi[2] = x_1, ...
def linear_regression_F(w: Array, xi: Array):
    y = xi[0]
    x = xi[1:]
    y_hat = jnp.inner(w, x)
    return L2_loss(y, y_hat)


def logistic_regression_F(w: Array, xi: Array):
    y = xi[0]
    x = xi[1:]
    y_hat = jax.nn.sigmoid(jnp.inner(w, x))
    return cross_entropy_loss(y, y_hat)


F_register = {
    'linear_regression_F': linear_regression_F,
    'logistic_regression_F': logistic_regression_F,
}
