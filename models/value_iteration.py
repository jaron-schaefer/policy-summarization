import numpy as np

def optimal_value(n_states, n_actions, transition_probabilities, reward, terminal_states,
                  discount, threshold=1e-2):
    """
    Calculate the optimal value function for a Markov Decision Process (MDP).

    :param n_states: Number of states in the MDP.
    :type n_states: int
    :param n_actions: Number of actions in the MDP.
    :type n_actions: int
    :param transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    :type transition_probabilities: np.array
    :param reward: Vector of rewards for each state.
    :type reward: numpy.ndarray
    :param terminal_states: List of indices of terminal states.
    :type terminal_states: list
    :param discount: MDP discount factor.
    :type discount: float
    :param threshold: Convergence threshold, default 1e-2.
    :type threshold: float
    :return: Array of values for each state.
    :rtype: numpy.ndarray
    """ 

    v = np.zeros(n_states)
    actions = range(n_actions)


    diff = float("inf")
    i = 0
    while (diff > threshold) & (i < 100):
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")

            if s in terminal_states:
                v[s] = 0
                continue

            # calculate the value of the current state based on all possible actions
            for a in actions:
                tp = transition_probabilities[s, a, :]
                
                max_v = max(max_v, np.dot(tp, reward + discount*v)) 

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
        i += 1
    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, terminal_states,
                discount, threshold=1e-2):
    """
    Calculates the optimal policy for a given MDP.

    :param n_states: number of states
    :param n_actions: number of actions
    :param transition_probabilities: transition probabilities of the MDP
    :param reward: reward function
    :param terminal_states: list of terminal states
    :param discount: discount factor
    :param threshold: threshold for value iteration
    :return: optimal policy, optimal value function
    """


    v = optimal_value(n_states, n_actions, transition_probabilities, reward, terminal_states, 
                      discount, threshold)

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if s in terminal_states:
            policy[s] = 0 #default to up
            continue
        max_v = float("-inf")
        for a in range(n_actions):
            tp = transition_probabilities[s, a, :]
            val = np.dot(tp, reward + discount * v)
            if val > max_v:
                max_v = val
                policy[s] = a

    return policy, v



def find_stochastic_policy(n_states, n_actions, transition_probabilities, reward, terminal_states, discount, threshold=1e-2):
    """
    Calculate the stochastic policy given the state-action values.
    
    :param n_states: number of states
    :param n_actions: number of actions
    :param transition_probabilities: transition probabilities of the environment
    :param reward: reward function
    :param terminal_states: terminal states
    :param discount: discount factor
    :param threshold: convergence threshold
    :return: stochastic policy
    """

    v = optimal_value(n_states, n_actions, transition_probabilities, reward, terminal_states, discount, threshold)
    
    actions = range(n_actions)
    # Get Q using equation 9.2 from Ziebart's thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):

        if i in terminal_states:
            # Set Q-values for terminal state to 0
            Q[i, :] = 0
        else:
            for j in actions:
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)

    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))

    return Q