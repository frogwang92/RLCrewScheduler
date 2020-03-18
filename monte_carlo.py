import gym
import numpy as np
import random
import sys
from collections import defaultdict
if "../" not in sys.path:
    sys.path.append("../")
from env import crew_and_jobs

env = crew_and_jobs.JobCrewsEnv()

observe_episodes_interval = 10


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    all_states = set()

    # The final value function
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % observe_episodes_interval == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        first_state_index = hash(tuple(state))
        for t in range(2000):
            # print("step " + str(t))
            state_index = hash(tuple(state))
            action = policy(state, env)
            next_state, reward, done, _ = env.step(action)
            episode.append((state_index, action, reward))
            if state_index not in all_states:
                all_states.add(state_index)
                returns_sum[state_index] = 0
                returns_count[state_index] = 0
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        # assume state will never duplicate in a episode
        first_occurence_idx = 0
        for state_index, action, reward in episode:
            # Find the first occurance of the state in the episode
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state_index] += G
            returns_count[state_index] += 1.0
            V[state_index] = returns_sum[state_index] / returns_count[state_index]
            first_occurence_idx += 1

        if i_episode % observe_episodes_interval == 0:
            print(V[first_state_index])
            env.plot()

    return V


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.full(env.action_space.n, -np.inf))
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n, env)
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % observe_episodes_interval == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        if i_episode % observe_episodes_interval == 0:
            policy = make_epsilon_greedy_policy(Q, 0, env.action_space.n, env)
        else:
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n, env)

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        first_state_index = hash(state.tobytes())
        second_state_index = -1
        for t in range(2000):
            state_index = hash(state.tobytes())
            probs = policy(state_index)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state_index, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        for state_index, action in sa_in_episode:
            sa_pair = (state_index, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state_index and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state_index][action] = returns_sum[sa_pair] / returns_count[sa_pair]

        # print(Q[first_state_index])
        if i_episode % observe_episodes_interval == 0:
            env.plot()

    return Q, policy


def make_epsilon_greedy_policy(Q, epsilon, nA, _env):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        actions = _env.observe_actions()    # all available actions
        totalA = actions * epsilon / np.sum(actions)    # all action spaces, same size as np jobs

        best_action = np.argmax(Q[observation])
        if Q[observation][best_action] < 0:
            best_action = np.argwhere(totalA > 0)[0][0]

        totalA[best_action] += (1.0 - epsilon)
        return totalA

    return policy_fn


def sample_policy(observation, _env):
    """
    A policy that is full random
    """
    # np_jobs, np_crew_pool, np_crew_resting_time, np_crew_start_working_time, current_job, next_new_crew = \
    #     _env.observe_current_pool()

    # find crews from resting pool
    actions = _env.observe_actions()
    _action = random.sample(actions, 1)[0]
    return _action


# mc_prediction(sample_policy, env, num_episodes=10000)
Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.1)

