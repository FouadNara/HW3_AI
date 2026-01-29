from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple, Optional
import numpy as np

# This matches the order of probabilities in the transition_function tuples
ACTION_ORDER = [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]

def value_iteration(mdp: MDP, U_init: List[List[float]], epsilon=10 ** (-3)):
    U = [row[:] for row in U_init]
    
    while True:
        U_new = [row[:] for row in U]
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                state = (r, c)
                if mdp.board[r][c] == "WALL":
                    U_new[r][c] = None
                    continue
                
                if state in mdp.terminal_states:
                    U_new[r][c] = float(mdp.get_reward(state))
                    continue
                
                action_values = []
                for action in Action:
                    expected_val = 0
                    probs = mdp.transition_function[action]
                    for i, prob in enumerate(probs):
                        if prob > 0:
                            next_s = mdp.step(state, ACTION_ORDER[i])
                            val = U[next_s[0]][next_s[1]]
                            expected_val += prob * (val if val is not None else 0)
                    action_values.append(expected_val)
                
                U_new[r][c] = float(mdp.get_reward(state)) + mdp.gamma * max(action_values)
                delta = max(delta, abs(U_new[r][c] - U[r][c]))
        
        U = U_new
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    return U

def get_policy(mdp: MDP, U: List[List[float]]):
    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            state = (r, c)
            if mdp.board[r][c] == "WALL" or state in mdp.terminal_states:
                policy[r][c] = None
                continue

            best_action = None
            max_val = float('-inf')
            
            for action in Action:
                expected_val = 0
                probs = mdp.transition_function[action]
                for i, prob in enumerate(probs):
                    if prob > 0:
                        next_s = mdp.step(state, ACTION_ORDER[i])
                        val = U[next_s[0]][next_s[1]]
                        expected_val += prob * (val if val is not None else 0)
                
                if expected_val > max_val:
                    max_val = expected_val
                    best_action = action
            
            policy[r][c] = best_action
    return policy

def policy_evaluation(mdp: MDP, policy: List[List[Optional[Action]]]):
    U = [[0.0 if mdp.board[r][c] != "WALL" else None for c in range(mdp.num_col)] for r in range(mdp.num_row)]
    
    while True:
        U_new = [row[:] for row in U]
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                state = (r, c)
                if U[r][c] is None: continue
                if state in mdp.terminal_states:
                    U_new[r][c] = float(mdp.get_reward(state))
                    continue
                
                chosen_action = policy[r][c]
                # Handle cases where the policy might contain strings or Action Enums
                if isinstance(chosen_action, str):
                    chosen_action = Action[chosen_action]
                
                probs = mdp.transition_function[chosen_action]
                expected_val = 0
                for i, prob in enumerate(probs):
                    if prob > 0:
                        next_s = mdp.step(state, ACTION_ORDER[i])
                        val = U[next_s[0]][next_s[1]]
                        expected_val += prob * (val if val is not None else 0)
                
                U_new[r][c] = float(mdp.get_reward(state)) + mdp.gamma * expected_val
                delta = max(delta, abs(U_new[r][c] - U[r][c]))
        
        U = U_new
        if delta < 10 ** (-3):
            break
    return U

def policy_iteration(mdp: MDP, policy_init: List[List[Optional[Action]]]):
    policy = policy_init
    while True:
        U = policy_evaluation(mdp, policy)
        new_policy = get_policy(mdp, U)
        if new_policy == policy:
            break
        policy = new_policy
    return policy

def mc_algorithm(
    sim: Simulator,
    num_episodes: int,
    gamma: float,
    num_rows: int = 3,
    num_cols: int = 4,
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
    policy: List[List[Optional[Action]]] = None,
):
    # Initialize returns and visit counts for all grid cells
    total_returns = np.zeros((num_rows, num_cols))
    counts = np.zeros((num_rows, num_cols))
    
    # Use sim.replay to retrieve episodes from the simulation results
    for episode_gen in sim.replay(num_episodes=num_episodes):
        episode = list(episode_gen) # List of (state, reward, action, actual_action)
        visited_in_episode = set()
        
        for i in range(len(episode)):
            state = episode[i][0]
            
            # First-visit Monte Carlo: only process the first time we see 'state' in this episode
            if state not in visited_in_episode:
                visited_in_episode.add(state)
                
                # Calculate G: sum of discounted rewards from step i to the end
                G = 0
                for t in range(i, len(episode)):
                    reward = episode[t][1]
                    G += (gamma ** (t - i)) * reward
                
                total_returns[state[0]][state[1]] += G
                counts[state[0]][state[1]] += 1

    # Final estimation of Utility V
    V = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
    for r in range(num_rows):
        for c in range(num_cols):
            if counts[r][c] > 0:
                # Use .item() to convert the NumPy scalar to a native Python float
                V[r][c] = (total_returns[r][c] / counts[r][c]).item()
            else:
                V[r][c] = 0.0
                
    return V