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

    # =========================

    return V
