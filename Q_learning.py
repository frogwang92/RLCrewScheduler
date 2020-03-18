import gym
import numpy as np
import random
import itertools
import matplotlib
import sys
from collections import defaultdict
if "../" not in sys.path:
    sys.path.append("../")
from lib import plotting
matplotlib.style.use('ggplot')
from env import crew_and_jobs

observe_episodes_interval = 100

env = crew_and_jobs.JobCrewsEnv()

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
        actions = _env.observe_actions()  # all available actions
        totalA = actions * epsilon / np.sum(actions)  # all action spaces, same size as np jobs

        best_action = np.argmax(Q[observation])
        if Q[observation][best_action] < 0:
            best_action = np.argwhere(totalA > 0)[0][0]

        totalA[best_action] += (1.0 - epsilon)
        return totalA

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=1, epsilon=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n, env)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % observe_episodes_interval == 0:
            policy = make_epsilon_greedy_policy(Q, 0, env.action_space.n, env)
        else:
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n, env)

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            state_index = hash(state.tobytes())
            # Take a step
            action_probs = policy(state_index)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state_index = hash(next_state.tobytes())
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state_index])
            td_target = reward + discount_factor * Q[next_state_index][best_next_action]
            td_delta = td_target - Q[state_index][action]
            Q[state_index][action] += alpha * td_delta

            if done:
                break

            state = next_state

        if (i_episode + 1) % observe_episodes_interval == 0:
            env.plot()

    return Q, stats


Q, stats = q_learning(env, 10000)

plotting.plot_episode_stats(stats)