import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# We want to minimize (w*X + b - y)^2

# Import a dataset
X, y = make_regression(n_features=3)
X, X_test, y, y_test = train_test_split(X, y)


# Model parameters: weights w and bias b
params = {
    'w': jnp.zeros(X.shape[1:]),
    'b': 0.
}

# Create a python function yielding the model evaluation
def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']

# Define the loss (objective) function
def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse

# Compute the gradient using JAX
grad_fn = jax.grad(loss_fn)

# Define a function to update the parameters
def update(params, grads, stepsize):
    # This is a practical JAX function that allows us to update all parameters in the dictionary in a simple way
    return jax.tree_map(lambda p, g: p - stepsize * g, params, grads)


# The main training loop: 50 iterations of steepest descent with stepsize 0.05
for _ in range(50):
    # We evaluate the loss on the test dataset
    loss = loss_fn(params, X_test, y_test)
    print(loss)

    # We optimize the loss on the training dataset
    grads = grad_fn(params, X, y)
    params = update(params, grads, 0.05)