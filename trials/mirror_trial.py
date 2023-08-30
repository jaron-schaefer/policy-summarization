import yaml
import os
from gridworld import Gridworld

def mirror_trial(trial_path):
    assert os.path.splitext(trial_path)[1] == ".yaml"
    

    with open(trial_path, 'r') as file:
            trial_data = yaml.safe_load(file)
        
    trial_data = trial_data

    gridworld = Gridworld(trial_data['GRIDWORLD_1'], trial_data['REWARDS'])

    max_x = gridworld.max_x
    max_y = gridworld.max_y

    trial_data['SUMMARY'] = mirror_summary(trial_data['SUMMARY'], max_x, max_y)

    trial_data['GRIDWORLD_1'] = mirror_gridworld(trial_data['GRIDWORLD_1'])
    trial_data['MODEL_PATH_GRIDWORLD_1'] = mirror_trajectory(trial_data['MODEL_PATH_GRIDWORLD_1'], max_x, max_y)

    start_state_indice = gridworld.state_identifiers.index(trial_data['START_STATE_GRIDWORLD_1'])
    mirrored_start_state_indice = mirror_start_state(start_state_indice, max_x, max_y)
    trial_data['START_STATE_GRIDWORLD_1'] = gridworld.state_identifiers[mirrored_start_state_indice]

    trial_data['POLICY_GRIDWORLD_1'] = mirror_policy(trial_data['POLICY_GRIDWORLD_1'], max_x, max_y)
    trial_data['OPTIMAL_PATH_GRIDWORLD_1'] = mirror_trajectory(trial_data['OPTIMAL_PATH_GRIDWORLD_1'], max_x, max_y)



    trial_data['GRIDWORLD_2'] = mirror_gridworld(trial_data['GRIDWORLD_2'])
    trial_data['MODEL_PATH_GRIDWORLD_2'] = mirror_trajectory(trial_data['MODEL_PATH_GRIDWORLD_2'], max_x, max_y)

    start_state_indice = gridworld.state_identifiers.index(trial_data['START_STATE_GRIDWORLD_2'])
    mirrored_start_state_indice = mirror_start_state(start_state_indice, max_x, max_y)
    trial_data['START_STATE_GRIDWORLD_2'] = gridworld.state_identifiers[mirrored_start_state_indice]

    trial_data['POLICY_GRIDWORLD_2'] = mirror_policy(trial_data['POLICY_GRIDWORLD_2'], max_x, max_y)
    trial_data['OPTIMAL_PATH_GRIDWORLD_2'] = mirror_trajectory(trial_data['OPTIMAL_PATH_GRIDWORLD_2'], max_x, max_y)

    with open(trial_path, 'w') as file:
        yaml.dump(trial_data, file)


def mirror_gridworld(grid_list):
    mirrored_gridworld = mirror_gridworld_vertical(grid_list)
    mirrored_gridworld = mirror_gridworld_horizontal(mirrored_gridworld)
    return mirrored_gridworld

def mirror_policy(policy, max_x, max_y):
    mirrored_policy = mirror_policy_vertical(policy, max_x, max_y)
    mirrored_policy = mirror_policy_horizontal(mirrored_policy, max_x, max_y)
    return mirrored_policy
    
def mirror_summary(summary, max_x, max_y):
    mirrored_summary = []
    for traj in summary:
        mirrored_summary.append(mirror_trajectory(traj,max_x, max_y))
    return mirrored_summary

def mirror_trajectory(trajectory, max_x, max_y):
    mirrored_trajectory = mirror_trajectory_vertical(trajectory, max_x, max_y)
    mirrored_trajectory = mirror_trajectory_horizontal(mirrored_trajectory, max_x, max_y)
    return mirrored_trajectory
    

def mirror_start_state(state, max_x, max_y):
    mirrored_start_state = mirror_state_vertical(state, max_x, max_y)
    mirrored_start_state = mirror_state_horizontal(mirrored_start_state, max_x, max_y)
    return mirrored_start_state
    

def mirror_gridworld_vertical(grid_list):
    # Splitting the strings in each element by whitespace
    gridworld = [row.split() for row in grid_list]

    # Mirroring each row vertically
    mirrored_gridworld = [row[::-1] for row in gridworld]

    # Combining the elements of each row into a string
    mirrored_grid_list = [' '.join(row) for row in mirrored_gridworld]

    return mirrored_grid_list


def mirror_gridworld_horizontal(grid_list):
    # Splitting the strings in each element by whitespace
    gridworld = [row.split() for row in grid_list]

    # Reversing the order of rows in the grid
    mirrored_gridworld = gridworld[::-1]

    # Combining the elements of each row into a string
    mirrored_grid_list = [' '.join(row) for row in mirrored_gridworld]

    return mirrored_grid_list


def mirror_state_horizontal(state, max_x, max_y):
    assert state < max_x * max_y

    # Calculate the row and column of the state in the grid
    col = state // max_y
    row = state - col * max_y

    # Compute the mirrored state 
    mirrored_state = state + ((max_y-1) - row * 2)

    return mirrored_state

def mirror_state_vertical(state, max_x, max_y):

    assert state < max_x * max_y
    # Calculate the row and column of the state in the grid
    
    col = state // max_y
    row = state - col * max_y

    # Compute the mirrored state 
    mirrored_state = state + (max_y *(max_x-1) - max_y * col*2)

    return mirrored_state

def mirror_policy_horizontal(policy, max_x, max_y):
    mirrored_policy = [0] * len(policy)

    # Iterate over each state-action pair in the policy
    for state, action in enumerate(policy):
        # Get the mirrored state using the mirror_state function
        mirrored_state = mirror_state_horizontal(state, max_x, max_y)
        # Assign the action corresponding to the mirrored state in the mirrored_policy list
        mirrored_policy[mirrored_state] = action

    return mirrored_policy

def mirror_policy_vertical(policy, max_x, max_y):
    mirrored_policy = [0] * len(policy)

    # Iterate over each state-action pair in the policy
    for state, action in enumerate(policy):
        # Get the mirrored state using the mirror_state function
        mirrored_state = mirror_state_vertical(state, max_x, max_y)
        # Assign the action corresponding to the mirrored state in the mirrored_policy list
        mirrored_policy[mirrored_state] = action

    return mirrored_policy


def mirror_action_vertical(action):

    if action == 2:
        return 3
    elif action == 3:
        return 2
    else:
        return action
    
def mirror_action_horizontal(action):

    if action == 0:
        return 1
    elif action == 1:
        return 0
    else:
        return action


def mirror_trajectory_vertical(trajectory, max_x, max_y):
    mirrored_trajectory = []

    for state, action in trajectory:
        new_state = mirror_state_vertical(state,max_x, max_y)
        new_action = mirror_action_vertical(action)
        mirrored_trajectory.append([new_state, new_action])

    return mirrored_trajectory


def mirror_trajectory_horizontal(trajectory, max_x, max_y):
    mirrored_trajectory = []

    for state, action in trajectory:
        new_state = mirror_state_horizontal(state,max_x, max_y)
        new_action = mirror_action_horizontal(action)
        mirrored_trajectory.append([new_state, new_action])

    return mirrored_trajectory
