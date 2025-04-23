import jax
import jax.numpy as jnp
import haiku as hk
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# We want to minimize (w*X + b - y)^2

# create our dataset
X, y = make_regression(n_features=3)
X, X_test, y, y_test = train_test_split(X, y)

# Let us try out both linear and function approximators, with a shallow and a deep NN
linear_fa = True

# Note that here we have a lot of freedom in how we define our approximator,
# as the two following options yield very different functions, 
# with different parameters structure and dimension, 
# but the rest of our code is independent of the approximator
if linear_fa:

    # Model parameters are defined implicitly in haiku
    def nn_fun(X):
        # Create a shallow NN composed of 1 linear layer with an output of dimension 1
        lin = hk.Linear(1)
        return lin(X).ravel()

    stepsize = 0.05

else:

    def nn_fun(X):
        # Create a deep NN with an output of dimension 1
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(16), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros),
            jnp.ravel
        ))
        return seq(X)

    # The nonlinear approximator makes our problem harder to solve:
    # we need a smaller stepsize to converge
    stepsize = 0.0005

# The next line of code is necessary to transform the python function such that it is an immutable object with specific properties
# without_apply_rng: needed to define our NN as a deterministic one
my_nn_ = hk.without_apply_rng(hk.transform(nn_fun))

# Initialize parameters: we need to define a random seed for JAX: this is useful to be able to reproduce our results
# Here the only place in which we need a random number generator is when we initialize the parameters just below the next line of code
rng = jax.random.PRNGKey(seed=1)
# The init method initializes the parameters with random values. Note that here we pass X as an input. 
# This is only used to deduce the input dimension, the value of X does not matter.
params = my_nn_.init(rng, X)

# Define our function 'my_nn' using the 'apply' method: conceptually, this could be done inside the init function, 
# but, as in some cases one might find it useful to have it separate, JAX has split these two functions
my_nn = my_nn_.apply

# From here on, everything is essentially as in the basic code

def loss_fn(params, X, y):
    err = my_nn(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)

def update(params, grads, stepsize):
    return jax.tree_map(lambda p, g: p - stepsize * g, params, grads)


for _ in range(200):
    loss = loss_fn(params, X_test, y_test)
    print(loss)

    grads = grad_fn(params, X, y)
    params = update(params, grads, stepsize)

