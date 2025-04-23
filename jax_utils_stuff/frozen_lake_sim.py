import gymnasium as gym
import jax
import jax.numpy as jnp
# import numpy as np
import haiku as hk
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

# Function that returns a vector containing all parameters
def tree_ravel(pytree):
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_leaves(pytree)])


# the MDP
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
nA = env.action_space.n
nS = env.observation_space.n
nEp = 10
epsilon = 0.1

# My policy as a lookup table
Pi = {}
Pi[0] = 1
Pi[1] = 0
Pi[2] = 1
Pi[3] = 0
Pi[4] = 1
Pi[5] = 1
Pi[6] = 1
Pi[7] = 1
Pi[8] = 2
Pi[9] = 2
Pi[10] = 1
Pi[11] = 1
Pi[12] = 2
Pi[13] = 2
Pi[14] = 2
Pi[15] = 0

# My policy as a function,
# params is useless now but will be necessary when solving the MDP
def pi(params,s,rng):
    # we implement an epsilon-greedy policy based on Pi
    # note that it will be more efficient to handle the epsilon greedy part outside the policy
    # but this implementation is provided as it illustrates useful functions

    # one-hot encoding of the state:
    # vector of 0 except the component relative to a
    # will not necessarily be useful here, but rather to define the Q function NN
    a_oh = jax.nn.one_hot(Pi[s],num_classes=nA)
    # Greedy action
    aG = (a_oh == 1)
    # Non-greedy actions
    aNG = (a_oh != 1)
    # Probability to pick a given non-greedy action
    ngP = 1./sum(aNG.astype(float)) * epsilon


    # jax.random.choice
    # picks an element from the second argument
    # with probability defined by p
    # and random seed rng
    return int(jax.random.choice(rng,jnp.arange(nA),p=aG*(1-epsilon)+aNG*ngP))

# Generate a random seed
rng = jax.random.PRNGKey(seed=1)
# Use the random seed to generate a sequence of random seeds
rngs = hk.PRNGSequence(rng)
# A new random seed is obtained by calling next(rngs)


# Simulate
for ep in tqdm(range(nEp)):

    # For each episod you need to reset the initial conditions,
    # this outputs an initial state
    s = env.reset()[0]

    # Run the episode at most until it reaches the maximum number of steps
    for t in range(env.spec.max_episode_steps):

        # Call the policy to generate the needed action
        a = pi([],s,next(rngs))

        # # Alternative epsilon-greedy implementation
        # if random.random() < epsilon:
        #     a = int(env.action_space.sample())
        # else:
        #     a = int(pi(params, s, next(rngs))) # In this case we would have: def pi(params,s,rng): return Pi[s]

        # Taking a step in the environment yields
        # the next state s1
        # instantaneous reward r
        # whether we reached a terminal state or not
        # some additional information
        s1, r, done, _, info = env.step(a)

        # This visualizes what happens in the MDP (optional of course)
        env.render()

        # Modify the reward with a small incentive to keep moving
        if jnp.array_equal(s1, s):
            r = -0.01

        if done:
            break

        # Update the state before proceeding
        s = s1




