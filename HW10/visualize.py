"""
Visualize learned policy from Q-Learning or Value Iteration
Shows the policy as arrows on a grid
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym


def default_Q_value():
    return 0.0


def default_V_value():
    return 0.0


def plot_Q_policy(Q_table, title="Q-Learning Policy"):
    """Plot policy derived from Q-table"""
    n_rows, n_cols = 4, 4

    # Action mapping for FrozenLake-v1
    action_to_arrow = {
        0: '←',  # LEFT
        1: '↓',  # DOWN
        2: '→',  # RIGHT
        3: '↑'   # UP
    }

    # Create policy grid
    policy_grid = np.empty((n_rows, n_cols), dtype=object)

    # For each state, find best action
    for state in range(n_rows * n_cols):
        row = state // n_cols
        col = state % n_cols
        q_values = [Q_table[(state, a)] for a in range(4)]
        best_action = np.argmax(q_values)
        policy_grid[row, col] = action_to_arrow[best_action]

    # FrozenLake map
    holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
    start = (0, 0)
    goal = (3, 3)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # Draw grid with colors
    for row in range(n_rows):
        for col in range(n_cols):
            y = n_rows - row - 1

            # Color cells
            if (row, col) == start:
                color = 'lightgreen'
            elif (row, col) == goal:
                color = 'lightblue'
            elif (row, col) in holes:
                color = 'lightcoral'
            else:
                color = 'white'

            rect = plt.Rectangle((col, y), 1, 1, facecolor=color,
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Add arrows (only for non-terminal states)
            if (row, col) not in holes and (row, col) != goal:
                ax.text(col + 0.5, y + 0.5, policy_grid[row, col],
                       ha='center', va='center', fontsize=40, fontweight='bold')

            # Add labels for special states
            if (row, col) == start:
                ax.text(col + 0.5, y + 0.2, 'START', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='darkgreen')
            elif (row, col) == goal:
                ax.text(col + 0.5, y + 0.5, 'GOAL', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='darkblue')
            elif (row, col) in holes:
                ax.text(col + 0.5, y + 0.5, 'HOLE', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='darkred')

    # Customize plot
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_V_policy(V_table, title="Value Iteration Policy"):
    """Plot policy derived from V-table"""
    n_rows, n_cols = 4, 4
    env = gym.make('FrozenLake-v1')

    # Action mapping for FrozenLake-v1
    action_to_arrow = {
        0: '←',  # LEFT
        1: '↓',  # DOWN
        2: '→',  # RIGHT
        3: '↑'   # UP
    }

    # Create policy grid
    policy_grid = np.empty((n_rows, n_cols), dtype=object)

    # For each state, find best action using one-step lookahead
    for state in range(n_rows * n_cols):
        row = state // n_cols
        col = state % n_cols

        best_action = 0
        best_value = float('-inf')

        for action in range(4):
            expected_value = 0
            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                expected_value += prob * (reward + 0.99 * V_table[next_state] * (1 - done))

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        policy_grid[row, col] = action_to_arrow[best_action]

    # FrozenLake map
    holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
    start = (0, 0)
    goal = (3, 3)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # Draw grid with colors
    for row in range(n_rows):
        for col in range(n_cols):
            y = n_rows - row - 1

            # Color cells
            if (row, col) == start:
                color = 'lightgreen'
            elif (row, col) == goal:
                color = 'lightblue'
            elif (row, col) in holes:
                color = 'lightcoral'
            else:
                color = 'white'

            rect = plt.Rectangle((col, y), 1, 1, facecolor=color,
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Add arrows (only for non-terminal states)
            if (row, col) not in holes and (row, col) != goal:
                ax.text(col + 0.5, y + 0.5, policy_grid[row, col],
                       ha='center', va='center', fontsize=40, fontweight='bold')

            # Add labels for special states
            if (row, col) == start:
                ax.text(col + 0.5, y + 0.2, 'START', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='darkgreen')
            elif (row, col) == goal:
                ax.text(col + 0.5, y + 0.5, 'GOAL', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='darkblue')
            elif (row, col) in holes:
                ax.text(col + 0.5, y + 0.5, 'HOLE', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='darkred')

    # Customize plot
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Policy Visualization")
    print("=" * 60)

    # Visualize Q-Learning Policy
    try:
        print("\n[1] Loading Q-Learning policy...")
        Q_table, epsilon = pickle.load(open('Q_TABLE_QLearning.pkl', 'rb'))
        plot_Q_policy(Q_table, "Q-Learning Policy on FrozenLake-v1")
    except FileNotFoundError:
        print("Error: Q_TABLE_QLearning.pkl not found")
    except Exception as e:
        print(f"Error loading Q-Learning policy: {e}")

    # Visualize Value Iteration Policy
    try:
        print("\n[2] Loading Value Iteration policy...")
        V_table = pickle.load(open('V_TABLE_ValueIteration.pkl', 'rb'))
        plot_V_policy(V_table, "Value Iteration Policy on FrozenLake-v1")
    except FileNotFoundError:
        print("Error: V_TABLE_ValueIteration.pkl not found")
    except Exception as e:
        print(f"Error loading Value Iteration policy: {e}")