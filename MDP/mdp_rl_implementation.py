from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    while True:
        U_new = {}
        for state in mdp.states:
            U_new[state] = mdp.get_reward(state) + mdp.gamma * max(
                [sum([mdp.transition_function[state][action][i] * U_init[i] for i in mdp.states]) for action in mdp.actions]
            )
        if max(abs(U_new[state] - U_init[state]) for state in mdp.states) < epsilon:
            break
        U_init = U_new
    # ========================
    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    for state in mdp.states:
        policy[state] = max(mdp.actions, key=lambda action: mdp.get_reward(state) + mdp.gamma * sum([mdp.transition_function[state][action][i] * U[i] for i in mdp.states]))

    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    U = None
    # TODO:
    # ====== YOUR CODE: ======
    while True:
        U_new = {}
        for state in mdp.states:
            U_new[state] = mdp.get_reward(state) + mdp.gamma * sum([mdp.transition_function[state][policy[state]][i] * U[i] for i in mdp.states])
        if max(abs(U_new[state] - U[state]) for state in mdp.states) < 10 ** (-3):
            break
        U = U_new
    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    policy = policy_init
    while True:
        U = policy_evaluation(mdp, policy)
        new_policy = {}
        for state in mdp.states:
            new_policy[state] = max(mdp.actions, key=lambda action: mdp.get_reward(state) + mdp.gamma * sum([mdp.transition_function[state][action][i] * U[i] for i in mdp.states]))
        if new_policy == policy:
            break
        policy = new_policy
    optimal_policy = policy
    # ========================
    return optimal_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return the utility of each state.

    V = None

    # ====== YOUR CODE: ======
    # Initialize returns and counts for each state
    total_returns = { (r, c): 0.0 for r in range(num_rows) for c in range(num_cols) }
    counts = { (r, c): 0 for r in range(num_rows) for c in range(num_cols) }
    
    # Identify wall locations to set them to None later [cite: 311]
    # We assume the simulator or MDP can provide wall info; if not, we filter by visits.
    
    for _ in range(num_episodes):
        episode = sim.run_episode(policy) # Episode: list of (state, reward, action, actual_action) [cite: 291]
        
        visited_in_episode = set()
        for i in range(len(episode)):
            state = episode[i][0]
            
            # First-visit check 
            if state not in visited_in_episode:
                visited_in_episode.add(state)
                
                # Calculate G (return): sum of discounted rewards from this step onwards
                G = 0
                for t in range(i, len(episode)):
                    reward = episode[t][1]
                    G += (gamma ** (t - i)) * reward
                
                total_returns[state] += G
                counts[state] += 1

    # Finalize Utility V
    V = {}
    for r in range(num_rows):
        for c in range(num_cols):
            state = (r, c)
            if counts[state] > 0:
                V[state] = total_returns[state] / counts[state]
            else:
                # If not visited, utility is 0; if it's a wall, logic in simulator usually omits it [cite: 312]
                V[state] = 0.0
    # =========================

    return V
