from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt

COLORS =[(1.0, 1.0, 1.0),  # - Weiß
        (0.0, 1.0, 0.0),  # A Grün
        (1.0, 0.0, 0.0),  # B Rot
        (0.0, 0.0, 1.0),  # C Blau
        (1.0, 1.0, 0.0),  # D Gelb
        (1.0, 0.0, 1.0),  # E Magenta
        (0.0, 1.0, 1.0),  # F Cyan
        (1.0, 0.5, 0.0),  # G Orange
        (0.5, 0.0, 0.5),  # H Lila
        (0.0, 0.5, 0.5)]   # I Türkis


def plot_world(gridworld):
<<<<<<< HEAD
    """
    Plots the gridworld environment.

    :param gridworld: The Gridworld object representing the environment.
    :type gridworld: Gridworld
    """
=======
>>>>>>> cedccd8 (add files)
    
    fig, ax = plt.subplots(figsize=(gridworld.max_x,gridworld.max_y))
    ax.set_xticks(np.arange(0, gridworld.max_x, 1))
    ax.set_yticks(np.arange(0, gridworld.max_y, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, gridworld.max_x])
    ax.set_ylim([0, gridworld.max_y])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='k', linestyle='-', linewidth=1)

    for i, feature in enumerate(gridworld.single_state_features):
        feature = feature.tolist()
        if feature[0] ==1:
            continue
        position = gridworld.state_identifiers[i]
        color = COLORS[feature.index(1)]
        ax.add_patch(patches.Rectangle(position, 1, 1, facecolor=color, alpha=0.3))

    for i in range(gridworld.max_x):
        ax.text(i+0.5, -0.5, str(i), ha='center', va='center')
    for j in range(gridworld.max_y):
        ax.text(-0.5, j+0.5, str(j), ha='center', va='center', rotation=90)

    plt.show()




def plot_policy(gridworld, policy, values = []):
<<<<<<< HEAD
    """
    Plots the policy for the gridworld environment.

    :param gridworld: The Gridworld object representing the environment.
    :type gridworld: Gridworld
    :param policy: The policy mapping states to chosen actions.
    :type policy: dict
    :param values: List of value estimates for each state, defaults to [].
    :type values: list, optional
    """
=======
>>>>>>> cedccd8 (add files)
    
    fig, ax = plt.subplots(figsize=(gridworld.max_x,gridworld.max_y))
    ax.set_xticks(np.arange(0, gridworld.max_x, 1))
    ax.set_yticks(np.arange(0, gridworld.max_y, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, gridworld.max_x])
    ax.set_ylim([0, gridworld.max_y])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='k', linestyle='-', linewidth=1)

    for i, feature in enumerate(gridworld.single_state_features):
        feature = feature.tolist()
        if feature[0] ==1:
            continue
        position = gridworld.state_identifiers[i]
        color = COLORS[feature.index(1)]
        ax.add_patch(patches.Rectangle(position, 1, 1, facecolor=color, alpha=0.3))

    for i in range(gridworld.max_x):
        ax.text(i+0.5, -0.5, str(i), ha='center', va='center')
    for j in range(gridworld.max_y):
        ax.text(-0.5, j+0.5, str(j), ha='center', va='center', rotation=90)
    for i, action in enumerate(policy): 
            position = gridworld.state_identifiers[i]
            x,y = position
            if i not in gridworld.terminal_indices: 
                if action == 1:
                    dx, dy = (0, -0.25)
                    x+= 0.5
                    y+= 0.75
                elif action == 0:
                    dx, dy = (0, 0.25)
                    x+= 0.5
                    y+= 0.25
                elif action == 2:
                    dx, dy = (-0.25, 0)
                    x+= 0.75
                    y+= 0.5
                elif action == 3:
                    dx, dy = (0.25, 0)
                    x+= 0.25
                    y+= 0.5
                
                ax.arrow(x, y, dx, dy, width=0.1, head_width=0.25, head_length=0.25, color='orange')
            
            if len(values) != 0:
                ab, ay = position
                ax.text(ab+0.5, ay+0.5, str(round(values[i],2)), ha='center', va='center')


    plt.show()

def _plot_trajectory(gridworld, trajectory, ax = None):
<<<<<<< HEAD
    """
    Plots a trajectory on the gridworld environment.

    :param gridworld: The Gridworld object representing the environment.
    :type gridworld: Gridworld
    :param trajectory: The trajectory to be plotted.
    :type trajectory: list of tuple (int, int)
    :param ax: Optional axis for the plot, defaults to None.
    :type ax: matplotlib axis, optional
    :return: The axis containing the plotted trajectory.
    :rtype: matplotlib axis
    """
=======
>>>>>>> cedccd8 (add files)

    if ax is None:
        fig, ax = plt.subplots(figsize=(gridworld.max_x,gridworld.max_y))
    
    ax.set_xticks(np.arange(0, gridworld.max_x, 1))
    ax.set_yticks(np.arange(0, gridworld.max_y, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, gridworld.max_x])
    ax.set_ylim([0, gridworld.max_y])
    ax.tick_params(axis='both', length=0)
    ax.grid(color='k', linestyle='-', linewidth=1)

    for i, feature in enumerate(gridworld.single_state_features):
        feature = feature.tolist()
        if feature[0] ==1:
            continue
        position = gridworld.state_identifiers[i]
        color = COLORS[feature.index(1)]
        ax.add_patch(patches.Rectangle(position, 1, 1, facecolor=color, alpha=0.3))

    for i in range(gridworld.max_x):
        ax.text(i+0.5, -0.5, str(i), ha='center', va='center')
    for j in range(gridworld.max_y):
        ax.text(-0.5, j+0.5, str(j), ha='center', va='center', rotation=90)

    for state, action in trajectory:
            position = gridworld.state_identifiers[state]
            if state in gridworld.terminal_indices:
                break
            x,y = position
            if action == 1:
                dx, dy = (0, -0.25)
                x+= 0.5
                y+= 0.75
            elif action == 0:
                dx, dy = (0, 0.25)
                x+= 0.5
                y+= 0.25
            elif action == 2:
                dx, dy = (-0.25, 0)
                x+= 0.75
                y+= 0.5
            elif action == 3:
                dx, dy = (0.25, 0)
                x+= 0.25
                y+= 0.5
            
            ax.arrow(x, y, dx, dy, width=0.1, head_width=0.25, head_length=0.25, color='orange')
    ax.set_aspect('equal')

    return ax



def plot_trajectories(gridworld, trajectories):
<<<<<<< HEAD
    """
    Plots multiple trajectories on the gridworld environment.

    :param gridworld: The Gridworld object representing the environment.
    :type gridworld: Gridworld
    :param trajectories: List of trajectories to be plotted.
    :type trajectories: list of list of tuple (int, int)
    """
=======
>>>>>>> cedccd8 (add files)
    fig, axs = plt.subplots(1, len(trajectories),  figsize=(len(trajectories)*gridworld.max_x, gridworld.max_y))
    
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i in range(len(trajectories)):
        axs[i].set_title('Trajectory ' + str(i+1))
        _plot_trajectory(gridworld, trajectories[i], ax=axs[i])
    plt.show()