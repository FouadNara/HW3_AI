from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

# Helper to map the transition probability tuple to Action enums
ACTION_ORDER = [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Initialize U as a 2D list to match mdp.print_utility expectations
    U = [row[:] for row in U_init]
    
    while True:
        U_new = [row[:] for row in U]
        delta = 0
        
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                state = (r, c)
                
                # Skip walls
                if mdp.board[r][c] == "WALL":
                    continue
                
                # Terminal states have no future utility
                if state in mdp.terminal_states:
                    U_new[r][c] = float(mdp.get_reward(state))
                    continue
                
                # Bellman Update: R(s) + gamma * max_a(sum(P(s'|s,a) * U(s')))
                action_values = []
                for action in Action:
                    expected_val = 0
                    probs = mdp.transition_function[action]
                    for i, prob in enumerate(probs):
                        if prob > 0:
                            actual_dir = ACTION_ORDER[i]
                            next_s = mdp.step(state, actual_dir)
                            expected_val += prob * U[next_s[0]][next_s[1]]
                    action_values.append(expected_val)
                
                U_new[r][c] = float(mdp.get_reward(state)) + mdp.gamma * max(action_values)
                delta = max(delta, abs(U_new[r][c] - U[r][c]))
        
        U = U_new
        # Convergence criteria for value iteration
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
            
    return U


def get_policy(mdp, U):
    # Initialize policy as a 2D grid
    policy = [["None" for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            state = (r, c)
            
            if mdp.board[r][c] == "WALL" or state in mdp.terminal_states:
                policy[r][c] = "None"
                continue

            best_action = None
            max_val = float('-inf')
            
            for action in Action:
                expected_val = 0
                probs = mdp.transition_function[action]
                for i, prob in enumerate(probs):
                    if prob > 0:
                        actual_dir = ACTION_ORDER[i]
                        next_s = mdp.step(state, actual_dir)
                        expected_val += prob * U[next_s[0]][next_s[1]]
                
                if expected_val > max_val:
                    max_val = expected_val
                    best_action = action.value # Use string value for printing
            
            policy[r][c] = best_action
    return policy


def policy_evaluation(mdp, policy):
    # Initialize U with rewards
    U = [[0.0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    
    while True:
        U_new = [row[:] for row in U]
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                state = (r, c)
                if mdp.board[r][c] == "WALL":
                    continue
                if state in mdp.terminal_states:
                    U_new[r][c] = float(mdp.get_reward(state))
                    continue
                
                # U(s) = R(s) + gamma * sum(P(s'|s, pi(s)) * U(s'))
                # policy[r][c] is the string value, convert back to Action enum
                chosen_action = Action(policy[r][c])
                probs = mdp.transition_function[chosen_action]
                expected_val = 0
                for i, prob in enumerate(probs):
                    if prob > 0:
                        actual_dir = ACTION_ORDER[i]
                        next_s = mdp.step(state, actual_dir)
                        expected_val += prob * U[next_s[0]][next_s[1]]
                
                U_new[r][c] = float(mdp.get_reward(state)) + mdp.gamma * expected_val
                delta = max(delta, abs(U_new[r][c] - U[r][c]))
        
        U = U_new
        if delta < 10 ** (-3):
            break
    return U


def policy_iteration(mdp, policy_init):
    policy = policy_init
    while True:
        U = policy_evaluation(mdp, policy)
        new_policy = get_policy(mdp, U)
        
        if new_policy == policy:
            break
        policy = new_policy
    return policy


def mc_algorithm(sim, num_episodes, gamma, num_rows=3, num_cols=4, 
                 actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT], policy=None):
    
    total_returns = np.zeros((num_rows, num_cols))
    counts = np.zeros((num_rows, num_cols))
    
    for _ in range(num_episodes):
        # episode is a list of (state, reward, action, actual_action)
        episode = sim.run_episode(policy)
        
        visited_states = set()
        for i in range(len(episode)):
            state = episode[i][0]
            
            # First-visit Monte Carlo
            if state not in visited_states:
                visited_states.add(state)
                
                # Calculate discounted return G
                G = 0
                for t in range(i, len(episode)):
                    reward = episode[t][1]
                    G += (gamma ** (t - i)) * reward
                
                total_returns[state[0]][state[1]] += G
                counts[state[0]][state[1]] += 1

    # Create 2D list for utility
    V = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
    for r in range(num_rows):
        for c in range(num_cols):
            if counts[r][c] > 0:
                V[r][c] = total_returns[r][c] / counts[r][c]
                
    return V